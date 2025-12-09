import argparse
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description='Evaluate LigandMPNN designed sequences')
    p.add_argument('--in_dirs', type=str, nargs='+', required=True,
                       help='LigandMPNN output directories containing seqs/*.fa files (multiple directories allowed)')
    p.add_argument('--reference_fasta', type=str, required=True,
                       help='Path to WT reference sequence FASTA file')
    p.add_argument('--pocket_tsv', type=str, required=True,
                       help='Path to TSV file containing pocket residue definitions')
    p.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    p.add_argument('--dir_labels', type=str, nargs='*',
                       help='Labels for directories (optional, defaults to directory names). Must match order of in_dirs.')
    return p.parse_args()

def load_reference_sequence(reference_fasta):
    """Load reference sequence from FASTA file."""
    try:
        records = list(SeqIO.parse(reference_fasta, "fasta"))
        if not records:
            raise ValueError(f"No sequences found in {reference_fasta}")
        return str(records[0].seq)
    except Exception as e:
        raise ValueError(f"Error loading reference sequence: {e}")

def collect_prediction_sequences(in_dir):
    """Collect all predicted sequences from seqs/*.fa files."""
    seqs_pattern = os.path.join(in_dir, "seqs", "*.fa")
    fasta_files = glob.glob(seqs_pattern)
    
    if not fasta_files:
        print(f"Warning: No FASTA files found in {in_dir}/seqs/")
        return []
    
    all_sequences = []
    
    for fasta_file in fasta_files:
        filename = Path(fasta_file).name
        try:
            records = list(SeqIO.parse(fasta_file, "fasta"))
            # Skip first sequence (design target), collect predictions
            for idx, record in enumerate(records[1:], 1):
                all_sequences.append({
                    'filename': filename,
                    'sequence_index': idx,
                    'sequence': str(record.seq)
                })
        except Exception as e:
            print(f"Warning: Error parsing {fasta_file}: {e}")
            continue
    
    return all_sequences

def load_pocket_residues_from_tsv(tsv_file):
    """
    Load pocket residues from TSV file.
    Returns a set of residue indices (0-based).
    """
    if not os.path.exists(tsv_file):
        raise FileNotFoundError(f"TSV file not found: {tsv_file}")
    
    print(f"Reading pocket residues from: {tsv_file}")
    
    all_pocket_residues = set()
    
    try:
        # Read TSV file
        df = pd.read_csv(tsv_file, sep='\t')
        
        print(f"TSV file contains {len(df)} entries")
        
        # Check required columns
        if 'pdb' not in df.columns or 'residues' not in df.columns:
            raise ValueError(f"TSV file must contain 'pdb' and 'residues' columns. Found: {list(df.columns)}")
        
        # Process each row
        for _, row in df.iterrows():
            pdb_name = row['pdb']
            residues_str = row['residues']
            
            if pd.isna(residues_str) or residues_str.strip() == '':
                print(f"  {pdb_name}: No residues specified")
                continue
            
            # Parse residues (format: A55,A56,B385, etc.)
            residue_list = [res.strip() for res in residues_str.split(',')]
            
            pdb_pocket_residues = set()
            for residue in residue_list:
                if residue:
                    # Extract chain and residue number
                    try:
                        chain = residue[0]
                        res_num = residue[1:]
                        
                        # Convert to 0-based index
                        # Note: This assumes residue numbers are 1-based and consecutive
                        # You might need to adjust this based on your specific numbering scheme
                        res_idx = int(res_num) - 1
                        if res_idx >= 0:
                            pdb_pocket_residues.add(res_idx)
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse residue '{residue}' from {pdb_name}")
                        continue
            
            all_pocket_residues.update(pdb_pocket_residues)
            print(f"  {pdb_name}: {len(pdb_pocket_residues)} pocket residues")
        
        print(f"Total unique pocket residues (union): {len(all_pocket_residues)}")
        if all_pocket_residues:
            print(f"Pocket residue range: {min(all_pocket_residues)} - {max(all_pocket_residues)}")
        
    except Exception as e:
        raise ValueError(f"Error reading TSV file {tsv_file}: {e}")
    
    return all_pocket_residues

def detect_pocket_residues(reference_seq, pocket_tsv):
    """
    Load pocket residues from TSV file and filter by sequence length.
    """
    print("Loading pocket residues from TSV file...")
    pocket_positions = load_pocket_residues_from_tsv(pocket_tsv)
    
    # Filter out positions that exceed the reference sequence length
    valid_pocket_positions = {pos for pos in pocket_positions if pos < len(reference_seq)}
    
    if len(valid_pocket_positions) < len(pocket_positions):
        removed_count = len(pocket_positions) - len(valid_pocket_positions)
        print(f"Warning: Removed {removed_count} pocket residue positions that exceed sequence length ({len(reference_seq)})")
        removed_positions = pocket_positions - valid_pocket_positions
        print(f"Removed positions: {sorted(removed_positions)}")
    
    return valid_pocket_positions

def calculate_sequence_similarity(seq1, seq2):
    """Calculate position-wise sequence similarity."""
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        seq1, seq2 = seq1[:min_len], seq2[:min_len]
    
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1) if len(seq1) > 0 else 0

def calculate_position_wise_similarity(sequences, reference_seq):
    """Calculate similarity at each position compared to reference."""
    if not sequences:
        return []
    
    seq_length = len(reference_seq)
    position_similarities = []
    
    for pos in range(seq_length):
        matches = 0
        total = 0
        
        for seq_data in sequences:
            seq = seq_data['sequence']
            if pos < len(seq):
                if seq[pos] == reference_seq[pos]:
                    matches += 1
                total += 1
        
        similarity = matches / total if total > 0 else 0
        position_similarities.append(similarity)
    
    return position_similarities

def calculate_position_wise_variability(sequences, reference_seq):
    """Calculate variability (change rate) at each position among predictions."""
    if not sequences:
        return []
    
    seq_length = len(reference_seq)
    position_variabilities = []
    
    for pos in range(seq_length):
        residues_at_pos = []
        
        for seq_data in sequences:
            seq = seq_data['sequence']
            if pos < len(seq):
                residues_at_pos.append(seq[pos])
        
        if not residues_at_pos:
            position_variabilities.append(0)
            continue
        
        # Calculate variability as 1 - (most frequent residue frequency)
        from collections import Counter
        residue_counts = Counter(residues_at_pos)
        most_common_freq = residue_counts.most_common(1)[0][1]
        variability = 1 - (most_common_freq / len(residues_at_pos))
        position_variabilities.append(variability)
    
    return position_variabilities

def create_similarity_plot(position_similarities, pocket_positions, reference_seq, output_dir, label):
    """Create line plot showing position-wise similarity to reference."""
    positions = list(range(len(position_similarities)))
    
    plt.figure(figsize=(15, 6))
    
    # Create base line plot
    plt.plot(positions, position_similarities, linewidth=1, alpha=0.7, color='blue')
    
    # Highlight pocket positions
    pocket_sims = [position_similarities[i] if i in pocket_positions else np.nan 
                   for i in positions]
    non_pocket_sims = [position_similarities[i] if i not in pocket_positions else np.nan 
                       for i in positions]
    
    plt.plot(positions, pocket_sims, 'ro', markersize=3, alpha=0.8, label='Pocket residues')
    plt.plot(positions, non_pocket_sims, 'bo', markersize=2, alpha=0.6, label='Non-pocket residues')
    
    plt.xlabel('Residue Position')
    plt.ylabel('Similarity to Reference')
    plt.title(f'Position-wise Similarity to WT Reference - {label}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create safe filename
    safe_label = "".join(c for c in label if c.isalnum() or c in (' ', '-', '_')).rstrip()
    plt.savefig(os.path.join(output_dir, f'similarity_to_reference_{safe_label}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_variability_plot(position_variabilities, pocket_positions, reference_seq, output_dir, label):
    """Create line plot showing position-wise variability among predictions."""
    positions = list(range(len(position_variabilities)))
    
    plt.figure(figsize=(15, 6))
    
    # Create base line plot
    plt.plot(positions, position_variabilities, linewidth=1, alpha=0.7, color='green')
    
    # Highlight pocket positions
    pocket_vars = [position_variabilities[i] if i in pocket_positions else np.nan 
                   for i in positions]
    non_pocket_vars = [position_variabilities[i] if i not in pocket_positions else np.nan 
                       for i in positions]
    
    plt.plot(positions, pocket_vars, 'ro', markersize=3, alpha=0.8, label='Pocket residues')
    plt.plot(positions, non_pocket_vars, 'go', markersize=2, alpha=0.6, label='Non-pocket residues')
    
    plt.xlabel('Residue Position')
    plt.ylabel('Variability Among Predictions')
    plt.title(f'Position-wise Variability in Predicted Sequences - {label}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create safe filename
    safe_label = "".join(c for c in label if c.isalnum() or c in (' ', '-', '_')).rstrip()
    plt.savefig(os.path.join(output_dir, f'prediction_variability_{safe_label}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_integrated_similarity_plot(all_similarities, pocket_positions, reference_seq, output_dir, labels):
    """Create integrated plot showing similarity across all directories."""
    positions = list(range(len(reference_seq)))
    
    plt.figure(figsize=(25, 8))
    
    distinct_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
    ]
    
    # Cycle through distinct colors if more directories than colors
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(all_similarities))]
    
    # Plot lines for each directory with distinct styles
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']  # Different line styles
    markers = ['o', 's', '^', 'v', 'D', 'h', 'p', '*']  # Different markers
    
    for i, (similarities, label) in enumerate(zip(all_similarities, labels)):
        if similarities:  # Only plot if we have data
            line_style = line_styles[i % len(line_styles)]
            marker_style = markers[i % len(markers)]
            
            plt.plot(positions[:len(similarities)], similarities, 
                    linewidth=3, alpha=0.8, color=colors[i], label=label,
                    linestyle=line_style, marker=marker_style, markersize=4, 
                    markevery=max(1, len(similarities)//50))  # Show markers every 50th point
    
    # Add pocket region highlighting with a more subtle color
    for pos in pocket_positions:
        if pos < len(positions):
            plt.axvline(x=pos, color='red', alpha=0.3, linewidth=1)
    
    # Create legend patches for pocket regions
    from matplotlib.patches import Patch
    legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=3, label=label,
                                 linestyle=line_styles[i % len(line_styles)]) 
                      for i, label in enumerate(labels)]
    legend_elements.append(Patch(facecolor='red', alpha=0.3, label='Pocket regions'))
    
    plt.xlabel('Residue Position', fontsize=12)
    plt.ylabel('Similarity to Reference', fontsize=12)
    plt.title('Integrated Position-wise Similarity to WT Reference', fontsize=14)
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'integrated_similarity_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_integrated_variability_plot(all_variabilities, pocket_positions, reference_seq, output_dir, labels):
    """Create integrated plot showing variability across all directories."""
    positions = list(range(len(reference_seq)))
    
    plt.figure(figsize=(25, 8))
    
    distinct_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
    ]
    
    # Cycle through distinct colors if more directories than colors
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(all_variabilities))]
    
    # Plot lines for each directory with distinct styles
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']  # Different line styles
    markers = ['o', 's', '^', 'v', 'D', 'h', 'p', '*']  # Different markers
    
    for i, (variabilities, label) in enumerate(zip(all_variabilities, labels)):
        if variabilities:  # Only plot if we have data
            line_style = line_styles[i % len(line_styles)]
            marker_style = markers[i % len(markers)]
            
            plt.plot(positions[:len(variabilities)], variabilities, 
                    linewidth=3, alpha=0.8, color=colors[i], label=label,
                    linestyle=line_style, marker=marker_style, markersize=4,
                    markevery=max(1, len(variabilities)//50))  # Show markers every 50th point
    
    # Add pocket region highlighting with a more subtle color
    for pos in pocket_positions:
        if pos < len(positions):
            plt.axvline(x=pos, color='red', alpha=0.3, linewidth=1)
    
    # Create legend patches for pocket regions
    from matplotlib.patches import Patch
    legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=3, label=label,
                                 linestyle=line_styles[i % len(line_styles)]) 
                      for i, label in enumerate(labels)]
    legend_elements.append(Patch(facecolor='red', alpha=0.3, label='Pocket regions'))
    
    plt.xlabel('Residue Position', fontsize=12)
    plt.ylabel('Variability Among Predictions', fontsize=12)
    plt.title('Integrated Position-wise Variability in Predicted Sequences', fontsize=14)
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'integrated_variability_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_comparison_plot(all_dir_metrics, all_variabilities, pocket_positions, output_dir, labels):
    """Create bar plots and box plots comparing summary statistics across directories."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Summary Statistics Comparison Across Directories', fontsize=16)
    
    # Define distinct colors for bars - same as line plots for consistency
    distinct_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
    ]
    
    bar_colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(labels))]
    
    # Prepare similarity data for distribution plots
    overall_sim_data = []
    pocket_sim_data = []
    non_pocket_sim_data = []
    
    for metrics in all_dir_metrics:
        if len(metrics) > 0:
            overall_sim_data.append(metrics['overall_similarity'].values)
            pocket_sim_data.append(metrics['pocket_similarity'].values)
            non_pocket_sim_data.append(metrics['non_pocket_similarity'].values)
        else:
            overall_sim_data.append([0])
            pocket_sim_data.append([0])
            non_pocket_sim_data.append([0])
    
    # Prepare variability data
    overall_var_data = []
    pocket_var_data = []
    non_pocket_var_data = []
    
    for variabilities in all_variabilities:
        if variabilities:
            # Overall variability
            overall_var_data.append(variabilities)
            
            # Pocket variability
            pocket_variability = [variabilities[pos] for pos in pocket_positions 
                                if pos < len(variabilities)]
            pocket_var_data.append(pocket_variability if pocket_variability else [0])
            
            # Non-pocket variability
            non_pocket_variability = [variabilities[pos] for pos in range(len(variabilities)) 
                                    if pos not in pocket_positions]
            non_pocket_var_data.append(non_pocket_variability if non_pocket_variability else [0])
        else:
            overall_var_data.append([0])
            pocket_var_data.append([0])
            non_pocket_var_data.append([0])
    
    # Plot 1: Overall similarity distribution (box plot)
    box_plot1 = axes[0,0].boxplot(overall_sim_data, labels=labels, patch_artist=True)
    axes[0,0].set_title('Overall Similarity Distribution', fontweight='bold')
    axes[0,0].set_ylabel('Similarity')
    axes[0,0].set_ylim(0, 1)
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3, axis='y')
    
    # Color each box with distinct colors and add median labels
    for i, (patch, data) in enumerate(zip(box_plot1['boxes'], overall_sim_data)):
        patch.set_facecolor(bar_colors[i])
        patch.set_alpha(0.8)
        # Add median value label
        if len(data) > 0 and not all(x == 0 for x in data):
            median_val = np.median(data)
            axes[0,0].text(i + 1, median_val + 0.05, f'{median_val:.3f}', 
                          ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Plot 2: Pocket vs Non-pocket similarity
    x = np.arange(len(labels))
    width = 0.35
    
    pocket_sim_means = [np.mean(data) if len(data) > 0 and not all(x == 0 for x in data) else 0 
                       for data in pocket_sim_data]
    non_pocket_sim_means = [np.mean(data) if len(data) > 0 and not all(x == 0 for x in data) else 0 
                           for data in non_pocket_sim_data]
    
    bars2a = axes[0,1].bar(x - width/2, pocket_sim_means, width, label='Pocket', 
                          color='#ff6b6b', alpha=0.8, edgecolor='black', linewidth=1)
    bars2b = axes[0,1].bar(x + width/2, non_pocket_sim_means, width, label='Non-pocket', 
                          color='#4ecdc4', alpha=0.8, edgecolor='black', linewidth=1)
    axes[0,1].set_title('Pocket vs Non-pocket Similarity', fontweight='bold')
    axes[0,1].set_ylabel('Similarity')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(labels, rotation=45)
    axes[0,1].set_ylim(0, 1)
    axes[0,1].legend(fontsize=10)
    axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels for pocket/non-pocket bars
    for bars, values in [(bars2a, pocket_sim_means), (bars2b, non_pocket_sim_means)]:
        for bar, val in zip(bars, values):
            if val > 0:
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                              f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 3: Overall variability distribution (box plot)
    box_plot3 = axes[1,0].boxplot(overall_var_data, labels=labels, patch_artist=True)
    axes[1,0].set_title('Overall Variability Distribution', fontweight='bold')
    axes[1,0].set_ylabel('Variability')
    axes[1,0].set_ylim(0, 1)
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3, axis='y')
    
    # Color each box with distinct colors and add median labels
    for i, (patch, data) in enumerate(zip(box_plot3['boxes'], overall_var_data)):
        patch.set_facecolor(bar_colors[i])
        patch.set_alpha(0.8)
        # Add median value label
        if len(data) > 0 and not all(x == 0 for x in data):
            median_val = np.median(data)
            axes[1,0].text(i + 1, median_val + 0.05, f'{median_val:.3f}', 
                          ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Plot 4: Pocket vs Non-pocket variability
    pocket_var_means = [np.mean(data) if len(data) > 0 and not all(x == 0 for x in data) else 0 
                       for data in pocket_var_data]
    non_pocket_var_means = [np.mean(data) if len(data) > 0 and not all(x == 0 for x in data) else 0 
                           for data in non_pocket_var_data]
    
    bars4a = axes[1,1].bar(x - width/2, pocket_var_means, width, label='Pocket', 
                          color='#ff6b6b', alpha=0.8, edgecolor='black', linewidth=1)
    bars4b = axes[1,1].bar(x + width/2, non_pocket_var_means, width, label='Non-pocket', 
                          color='#4ecdc4', alpha=0.8, edgecolor='black', linewidth=1)
    axes[1,1].set_title('Pocket vs Non-pocket Variability', fontweight='bold')
    axes[1,1].set_ylabel('Variability')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(labels, rotation=45)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].legend(fontsize=10)
    axes[1,1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels for variability bars
    for bars, values in [(bars4a, pocket_var_means), (bars4b, non_pocket_var_means)]:
        for bar, val in zip(bars, values):
            if val > 0:
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                              f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_integrated_similarity_plot(all_similarities, pocket_positions, reference_seq, output_dir, labels):
    """Create integrated plot showing similarity across all directories."""
    positions = list(range(len(reference_seq)))
    
    plt.figure(figsize=(25, 8))
    
    distinct_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
    ]
    
    # Cycle through distinct colors if more directories than colors
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(all_similarities))]
    
    # Plot lines for each directory - clean lines only
    for i, (similarities, label) in enumerate(zip(all_similarities, labels)):
        if similarities:  # Only plot if we have data
            plt.plot(positions[:len(similarities)], similarities, 
                    linewidth=3, alpha=0.8, color=colors[i], label=label)
    
    # Add pocket region highlighting with a more subtle color
    for pos in pocket_positions:
        if pos < len(positions):
            plt.axvline(x=pos, color='red', alpha=0.3, linewidth=2)
    
    # Create legend patches for pocket regions
    from matplotlib.patches import Patch
    legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=3, label=label) 
                      for i, label in enumerate(labels) if all_similarities[i]]
    legend_elements.append(Patch(facecolor='red', alpha=0.3, label='Pocket regions'))
    
    plt.xlabel('Residue Position', fontsize=12)
    plt.ylabel('Similarity to Reference', fontsize=12)
    plt.title('Integrated Position-wise Similarity to WT Reference', fontsize=14)
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'integrated_similarity_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_integrated_variability_plot(all_variabilities, pocket_positions, reference_seq, output_dir, labels):
    """Create integrated plot showing variability across all directories."""
    positions = list(range(len(reference_seq)))
    
    plt.figure(figsize=(25, 8))
    
    distinct_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
    ]
    
    # Cycle through distinct colors if more directories than colors
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(all_variabilities))]
    
    # Plot lines for each directory - clean lines only
    for i, (variabilities, label) in enumerate(zip(all_variabilities, labels)):
        if variabilities:  # Only plot if we have datas
            plt.plot(positions[:len(variabilities)], variabilities, 
                    linewidth=3, alpha=0.8, color=colors[i], label=label)
    
    # Add pocket region highlighting with a more subtle color
    for pos in pocket_positions:
        if pos < len(positions):
            plt.axvline(x=pos, color='red', alpha=0.3, linewidth=2)
    
    # Create legend patches for pocket regions
    from matplotlib.patches import Patch
    legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=3, label=label) 
                      for i, label in enumerate(labels) if all_variabilities[i]]
    legend_elements.append(Patch(facecolor='red', alpha=0.3, label='Pocket regions'))
    
    plt.xlabel('Residue Position', fontsize=12)
    plt.ylabel('Variability Among Predictions', fontsize=12)
    plt.title('Integrated Position-wise Variability in Predicted Sequences', fontsize=14)
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'integrated_variability_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def calculate_regional_metrics(sequences, reference_seq, pocket_positions):
    """Calculate pocket and non-pocket region metrics for each sequence."""
    results = []
    
    for seq_data in sequences:
        seq = seq_data['sequence']
        
        # Overall similarity
        overall_similarity = calculate_sequence_similarity(seq, reference_seq)
        
        # Pocket region similarity
        pocket_matches = 0
        pocket_total = 0
        for pos in pocket_positions:
            if pos < len(seq) and pos < len(reference_seq):
                if seq[pos] == reference_seq[pos]:
                    pocket_matches += 1
                pocket_total += 1
        pocket_similarity = pocket_matches / pocket_total if pocket_total > 0 else 0
        
        # Non-pocket region similarity
        non_pocket_matches = 0
        non_pocket_total = 0
        for pos in range(min(len(seq), len(reference_seq))):
            if pos not in pocket_positions:
                if seq[pos] == reference_seq[pos]:
                    non_pocket_matches += 1
                non_pocket_total += 1
        non_pocket_similarity = non_pocket_matches / non_pocket_total if non_pocket_total > 0 else 0
        
        results.append({
            'filename': seq_data['filename'],
            'sequence_index': seq_data['sequence_index'],
            'sequence': seq,
            'overall_similarity': overall_similarity,
            'pocket_similarity': pocket_similarity,
            'non_pocket_similarity': non_pocket_similarity
        })
    
    return results

# additional plot takes target as a baseline for comparison
def create_baseline_difference_plot(all_variabilities, pocket_positions, reference_seq, output_dir, labels, baseline_label="tar"):
    """
    Create plot showing the DIFFERENCE in variability between modified methods and baseline.
    Positive values = higher variability than baseline, Negative values = lower variability than baseline.
    """
    positions = list(range(len(reference_seq)))
    
    # Find the baseline method
    baseline_idx = None
    for i, label in enumerate(labels):
        if baseline_label.lower() in label.lower():
            baseline_idx = i
            break
    
    if baseline_idx is None:
        print(f"Warning: Baseline method '{baseline_label}' not found in labels: {labels}")
        return
    
    baseline_variabilities = all_variabilities[baseline_idx]
    if not baseline_variabilities:
        print(f"Warning: No variability data found for baseline method '{labels[baseline_idx]}'")
        return
    
    plt.figure(figsize=(25, 10))
    
    # Define distinct colors for modified methods
    distinct_colors = [
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
    ]
    
    # Add horizontal reference line at y=0 (no difference from baseline)
    plt.axhline(y=0, color='#1f77b4', linewidth=3, alpha=0.8, 
                label=f'{labels[baseline_idx]} (baseline)', zorder=1)
    
    # Plot relative differences for each modified method
    color_idx = 0
    max_abs_diff = 0  # Track maximum absolute difference for y-axis scaling
    
    for i, (variabilities, label) in enumerate(zip(all_variabilities, labels)):
        if i == baseline_idx or not variabilities:
            continue  # Skip baseline or empty data
        
        # Calculate difference from baseline
        min_length = min(len(variabilities), len(baseline_variabilities))
        difference = []
        
        for pos in range(min_length):
            diff = variabilities[pos] - baseline_variabilities[pos]
            difference.append(diff)
            max_abs_diff = max(max_abs_diff, abs(diff))
        
        # Plot the difference
        color = distinct_colors[color_idx % len(distinct_colors)]
        plt.plot(positions[:len(difference)], difference, 
                linewidth=3, alpha=0.8, color=color, 
                label=f'{label}', zorder=2)
        
        color_idx += 1
    
    # Add pocket region highlighting
    for pos in pocket_positions:
        if pos < len(positions):
            plt.axvline(x=pos, color='red', alpha=0.3, linewidth=2)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = []
    
    # Add baseline reference
    legend_elements.append(plt.Line2D([0], [0], color='#1f77b4', lw=3, 
                                     label=f'{labels[baseline_idx]} (baseline = 0)'))
    
    # Add modified methods
    color_idx = 0
    for i, label in enumerate(labels):
        if i == baseline_idx or not all_variabilities[i]:
            continue
        color = distinct_colors[color_idx % len(distinct_colors)]
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=3, label=label))
        color_idx += 1
    
    # Add pocket regions
    legend_elements.append(Patch(facecolor='red', alpha=0.3, label='Pocket regions'))
    
    plt.xlabel('Residue Position', fontsize=12)
    plt.ylabel(f'Variability Difference from {labels[baseline_idx]}', fontsize=12)
    plt.title(f'Variability Difference: Modified Methods vs {labels[baseline_idx]} (Baseline)', fontsize=14)
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits based on data range
    if max_abs_diff > 0:
        y_margin = max_abs_diff * 0.1
        plt.ylim(-max_abs_diff - y_margin, max_abs_diff + y_margin)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'baseline_difference_comparison_{baseline_label}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Baseline difference plot saved: baseline_difference_comparison_{baseline_label}.png")

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate labels for directories (matched by order)
    if args.dir_labels:
        if len(args.dir_labels) != len(args.in_dirs):
            raise ValueError(f"Number of dir_labels ({len(args.dir_labels)}) must match number of in_dirs ({len(args.in_dirs)})")
        labels = args.dir_labels
    else:
        labels = [Path(d).name for d in args.in_dirs]
    
    print(f"Processing {len(args.in_dirs)} directories:")
    for i, (in_dir, label) in enumerate(zip(args.in_dirs, labels)):
        print(f"  {i+1}. {in_dir} -> '{label}'")
    
    print(f"\nLoading reference sequence from {args.reference_fasta}")
    reference_seq = load_reference_sequence(args.reference_fasta)
    print(f"Reference sequence length: {len(reference_seq)}")
    
    # Load pocket residues from TSV file (used for ALL directories)
    print(f"\nLoading pocket residues from {args.pocket_tsv}")
    pocket_positions = detect_pocket_residues(reference_seq, args.pocket_tsv)
    print(f"Final pocket positions: {len(pocket_positions)} residues")
    if pocket_positions:
        sorted_positions = sorted(list(pocket_positions))
        print(f"Pocket residue positions: {sorted_positions}")
    
    # Process each directory
    all_similarities = []
    all_variabilities = []
    all_dir_metrics = []
    
    for i, (in_dir, label) in enumerate(zip(args.in_dirs, labels)):
        print(f"\n=== Processing directory {i+1}/{len(args.in_dirs)}: '{label}' ===")
        
        # Collect sequences
        print(f"Collecting prediction sequences from {in_dir}")
        sequences = collect_prediction_sequences(in_dir)
        print(f"Found {len(sequences)} predicted sequences")
        
        if not sequences:
            print(f"No sequences found in {in_dir}, skipping...")
            all_similarities.append([])
            all_variabilities.append([])
            all_dir_metrics.append(pd.DataFrame())
            continue
        
        # Calculate metrics (using the same pocket positions for all directories)
        print("Calculating position-wise similarity to reference...")
        position_similarities = calculate_position_wise_similarity(sequences, reference_seq)
        all_similarities.append(position_similarities)
        
        print("Calculating position-wise variability among predictions...")
        position_variabilities = calculate_position_wise_variability(sequences, reference_seq)
        all_variabilities.append(position_variabilities)
        
        # Create individual plots
        print("Creating individual visualization plots...")
        create_similarity_plot(position_similarities, pocket_positions, reference_seq, args.output_dir, label)
        create_variability_plot(position_variabilities, pocket_positions, reference_seq, args.output_dir, label)
        
        # Calculate regional metrics
        print("Calculating regional metrics...")
        regional_metrics = calculate_regional_metrics(sequences, reference_seq, pocket_positions)
        dir_metrics_df = pd.DataFrame(regional_metrics)
        dir_metrics_df['directory'] = label
        all_dir_metrics.append(dir_metrics_df)
        
        # Save individual CSV
        individual_csv_path = os.path.join(args.output_dir, f'sequence_evaluation_{label}.csv')
        dir_metrics_df.to_csv(individual_csv_path, index=False)
        print(f"Individual results saved to {individual_csv_path}")
        
        # Print individual summary
        if len(dir_metrics_df) > 0:
            print(f"\nSummary for '{label}':")
            print(f"  Average overall similarity: {dir_metrics_df['overall_similarity'].mean():.3f}")
            print(f"  Average pocket similarity: {dir_metrics_df['pocket_similarity'].mean():.3f}")
            print(f"  Average non-pocket similarity: {dir_metrics_df['non_pocket_similarity'].mean():.3f}")
            
            if position_variabilities:
                avg_variability = np.mean(position_variabilities)
                if pocket_positions:
                    pocket_variability = np.mean([position_variabilities[pos] for pos in pocket_positions 
                                                if pos < len(position_variabilities)])
                    non_pocket_variability = np.mean([position_variabilities[pos] for pos in range(len(position_variabilities)) 
                                                    if pos not in pocket_positions])
                    print(f"  Average position variability: {avg_variability:.3f}")
                    print(f"  Average pocket variability: {pocket_variability:.3f}")
                    print(f"  Average non-pocket variability: {non_pocket_variability:.3f}")
    
    # Create integrated plots
    print("\n=== Creating integrated comparison plots ===")
    create_integrated_similarity_plot(all_similarities, pocket_positions, reference_seq, args.output_dir, labels)
    create_integrated_variability_plot(all_variabilities, pocket_positions, reference_seq, args.output_dir, labels)
    create_summary_comparison_plot(all_dir_metrics, all_variabilities, pocket_positions, args.output_dir, labels)

    print("\n=== Creating baseline comparison plots ===")
    if "tar" in args.dir_labels and "mod" in args.dir_labels:
        print("Baseline 'tar' and modified 'mod' methods detected in labels.")
        create_baseline_difference_plot(all_variabilities, pocket_positions, reference_seq, args.output_dir, labels)

    # Save combined CSV
    if all_dir_metrics and any(len(df) > 0 for df in all_dir_metrics):
        combined_df = pd.concat([df for df in all_dir_metrics if len(df) > 0], ignore_index=True)
        combined_csv_path = os.path.join(args.output_dir, 'combined_sequence_evaluation_results.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined results saved to {combined_csv_path}")
        
        # Print overall summary
        print("\n=== Overall Summary ===")
        for label in labels:
            subset = combined_df[combined_df['directory'] == label]
            if len(subset) > 0:
                print(f"'{label}': {len(subset)} sequences, "
                      f"avg similarity: {subset['overall_similarity'].mean():.3f}")
    
    print(f"\nAll results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()