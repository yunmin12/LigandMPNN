XML_BSITE_REPACK_MIN = """
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="scorefxn_full" weights="ref2015">
      <Reweight scoretype="coordinate_constraint" weight="0.5"/>
      <Reweight scoretype="atom_pair_constraint" weight="0.3" />
    </ScoreFunction>
    <ScoreFunction name="scorefxn_soft" weights="ref2015_soft">
      <Reweight scoretype="coordinate_constraint" weight="1.0"/>
      <Reweight scoretype="fa_rep" weight="0.2"/>
      <Reweight scoretype="atom_pair_constraint" weight="0.5" />
    </ScoreFunction>
  </SCOREFXNS>

  <RESIDUE_SELECTORS>
    <Index name="repack_res" resnums="{0}"/>

    <Not name="fix_res" selector="repack_res"/>
    <Chain name="chB" chains="B"/>
  </RESIDUE_SELECTORS>

  <TASKOPERATIONS>
    <OperateOnResidueSubset name="dont_repack" selector="fix_res">
      <PreventRepackingRLT/>
    </OperateOnResidueSubset>
    <OperateOnResidueSubset name="repack_selected_res" selector="repack_res">
      <RestrictToRepackingRLT/>
    </OperateOnResidueSubset>
    #
    <InitializeFromCommandline name="init"/>
    <ExtraRotamersGeneric name="ex1ex2" ex1="1" ex2="1"/>
  </TASKOPERATIONS>

  <FILTERS>
    <ScoreType name="totalscore" scorefxn="scorefxn_full" threshold="9999999" confidence="1"/>
    <ResidueCount name="nres" confidence="1" />
    <CalculatorFilter name="res_totalscore" confidence="1" equation="SCORE/NRES" threshold="9999999">
      <Var name="SCORE" filter_name="totalscore" />
      <Var name="NRES" filter_name="nres" />
    </CalculatorFilter>


    <Ddg name="ddg" threshold="9999999" jump="1" repeats="1" repack="1" repack_bound="true" repack_unbound="true" relax_unbound="false" scorefxn="scorefxn_full"/>
    ContactMolecularSurface name="cms" target_selector="chB" binder_selector="repack_res"/>
    ContactMolecularSurface name="cms" target_selector="ligres" binder_selector="repack_res" use_rosetta_radii="true"/>

  </FILTERS>

  <MOVERS>
    <AddConstraints name="add_ca_csts" >
      <CoordinateConstraintGenerator name="coord_cst_gen" ca_only="true" native="true" align_reference="true"/>
    </AddConstraints>
    <ConstraintSetMover name="add_lig_d_csts" add_constraints="1" cst_file="{1}" />
    <ConstraintSetMover name="rm_lig_d_csts" cst_file="none" />
    <RemoveConstraints name="rm_csts" constraint_generators="coord_cst_gen"/>
    <PackRotamersMover name="repack" scorefxn="scorefxn_full" task_operations="init,ex1ex2,repack_selected_res,dont_repack"/>
    <PackRotamersMover name="repack_soft" scorefxn="scorefxn_soft" task_operations="init,ex1ex2,repack_selected_res,dont_repack"/>
     MinMover name="min_full" bb="1" chi="1" jump="0" scorefxn="scorefxn_full"/>
    <MinMover name="min_full" bb="1" chi="1" jump="ALL" scorefxn="scorefxn_full"/>
    <MinMover name="min_soft" bb="1" chi="1" jump="ALL" scorefxn="scorefxn_soft"/>
  </MOVERS>

  <PROTOCOLS>
    <Add mover="add_ca_csts"/>
    <Add mover="add_lig_d_csts"/>
    <Add mover="repack_soft"/>
    <Add mover="min_soft"/>
    <Add mover="repack"/>
    <Add mover="min_full"/>
    <Add mover="rm_csts"/>
    <Add mover="rm_lig_d_csts"/>
    <Add filter="ddg"/>
    <Add filter="cms"/>
    <Add filter="res_totalscore"/>
    <Add filter="totalscore"/>
  </PROTOCOLS>

</ROSETTASCRIPTS>

"""

XML_BSITE_REPACK_MIN_BETA = """
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <!-- full: 살짝만 묶어둔 상태에서 최종 minimize -->
    <ScoreFunction name="scorefxn_full" weights="beta_genpot">
      <Reweight scoretype="coordinate_constraint" weight="0.3"/>
    </ScoreFunction>

    <!-- soft: repack + 초기 minimize용 -->
    <ScoreFunction name="scorefxn_soft" weights="beta_genpot_soft">
      <Reweight scoretype="coordinate_constraint" weight="0.5"/>
      <Reweight scoretype="fa_rep" weight="0.2"/>
    </ScoreFunction>
  </SCOREFXNS>

  <RESIDUE_SELECTORS>
    <!-- repack할 residue 리스트 (python에서 {0} 자리에 들어감) -->
    <Index name="repack_res" resnums="{0}"/>

    <Not name="fix_res" selector="repack_res"/>
    <Chain name="chB" chains="B"/>
  </RESIDUE_SELECTORS>

  <TASKOPERATIONS>
    <OperateOnResidueSubset name="dont_repack" selector="fix_res">
      <PreventRepackingRLT/>
    </OperateOnResidueSubset>

    <OperateOnResidueSubset name="repack_selected_res" selector="repack_res">
      <RestrictToRepackingRLT/>
    </OperateOnResidueSubset>

    <InitializeFromCommandline name="init"/>
    <ExtraRotamersGeneric name="ex1ex2" ex1="1" ex2="1"/>
  </TASKOPERATIONS>

  <FILTERS>
    <ScoreType name="totalscore" scorefxn="scorefxn_full" threshold="9999999"/>
    <ScoreType name="fa_rep"     scorefxn="scorefxn_full" score_type="fa_rep" threshold="9999999"/>

    <ResidueCount name="nres" />
    <CalculatorFilter name="res_totalscore" equation="SCORE/NRES" threshold="9999999">
      <Var name="SCORE" filter_name="totalscore" />
      <Var name="NRES"  filter_name="nres" />
    </CalculatorFilter>
  </FILTERS>

  <MOVERS>
    <!-- CA-only coordinate constraints: 전체 fold를 원본에 align한 상태로 약하게 묶기 -->
    <AddConstraints name="add_ca_csts">
      <CoordinateConstraintGenerator
          name="coord_cst_gen"
          ca_only="true"
          native="true"
          align_reference="true"/>
    </AddConstraints>

    <PackRotamersMover
        name="repack_soft"
        scorefxn="scorefxn_soft"
        task_operations="init,ex1ex2,repack_selected_res,dont_repack"/>

    <PackRotamersMover
        name="repack"
        scorefxn="scorefxn_full"
        task_operations="init,ex1ex2,repack_selected_res,dont_repack"/>

    <MinMover name="min_soft" bb="1" chi="1" jump="ALL" scorefxn="scorefxn_soft"/>
    <MinMover name="min_full" bb="1" chi="1" jump="ALL" scorefxn="scorefxn_full"/>

    <!-- 마지막에 coord constraints 제거하고 깨끗한 pose 남기고 싶으면 유지 -->
    <RemoveConstraints name="rm_csts" constraint_generators="coord_cst_gen"/>
  </MOVERS>

  <PROTOCOLS>
    <Add mover_name="add_ca_csts"/>
    <Add mover_name="repack_soft"/>
    <Add mover_name="min_soft"/>
    <Add mover_name="repack"/>
    <Add mover_name="min_full"/>
    <Add mover_name="rm_csts"/>
    <Add filter_name="fa_rep"/>
    <Add filter_name="res_totalscore"/>
    <Add filter_name="totalscore"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
"""

XML_BSITE_FASTRELAX = """
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="scorefxn_full" weights="beta_genpot">
      <Reweight scoretype="coordinate_constraint" weight="0.5"/>
      <Reweight scoretype="atom_pair_constraint" weight="0.5" />
    </ScoreFunction>
    <ScoreFunction name="scorefxn_soft" weights="beta_genpot_soft">
      <Reweight scoretype="coordinate_constraint" weight="1.0"/>
      <Reweight scoretype="fa_rep" weight="0.2"/>
      <Reweight scoretype="atom_pair_constraint" weight="0.5" />
    </ScoreFunction>
    <ScoreFunction name="scorefxn_nocst" weights="beta_genpot">
      <Reweight scoretype="coordinate_constraint" weight="0.0"/>
      <Reweight scoretype="atom_pair_constraint" weight="0.0" />
    </ScoreFunction>
  </SCOREFXNS>

  <RESIDUE_SELECTORS>
    <Index name="repack_res" resnums="{0}"/>
    <Not name="fix_res" selector="repack_res"/>
    <Chain name="chB" chains="B"/>
  </RESIDUE_SELECTORS>

  <TASKOPERATIONS>
    <OperateOnResidueSubset name="dont_repack" selector="fix_res">
      <PreventRepackingRLT/>
    </OperateOnResidueSubset>
    <OperateOnResidueSubset name="repack_selected_res" selector="repack_res">
      <RestrictToRepackingRLT/>
    </OperateOnResidueSubset>
    #
    <InitializeFromCommandline name="init"/>
    <ExtraRotamersGeneric name="ex1ex2" ex1="1" ex2="1"/>
  </TASKOPERATIONS>

  <FILTERS>
    <ScoreType name="totalscore" scorefxn="scorefxn_full" threshold="9999999" confidence="1"/>
    <ResidueCount name="nres" confidence="1" />
    <CalculatorFilter name="res_totalscore" confidence="1" equation="SCORE/NRES" threshold="999999">
      <Var name="SCORE" filter_name="totalscore" />
      <Var name="NRES" filter_name="nres" />
    </CalculatorFilter>

    <Ddg name="ddg_after_relax_cst" threshold="99999" jump="1" repeats="1" repack="1" repack_bound="true" repack_unbound="true" relax_unbound="false" scorefxn="scorefxn_full"/>
    <Ddg name="ddg" threshold="99999" jump="1" repeats="1" repack="1" repack_bound="true" repack_unbound="true" relax_unbound="false" scorefxn="scorefxn_nocst"/>
    ContactMolecularSurface name="cms" target_selector="chB" binder_selector="repack_res"/>
    <ContactMolecularSurface name="cms" target_selector="chB" binder_selector="repack_res" use_rosetta_radii="true"/>

  </FILTERS>

  <MOVERS>
    <AddConstraints name="add_ca_csts" >
      <CoordinateConstraintGenerator name="coord_cst_gen" ca_only="true" native="true" align_reference="true"/>
    </AddConstraints>
    <ConstraintSetMover name="add_lig_d_csts" add_constraints="1" cst_file="{1}"/>
    <ConstraintSetMover name="rm_lig_d_csts" cst_file="none"/>
    <RemoveConstraints name="rm_csts" constraint_generators="coord_cst_gen"/>

    <FastRelax name="fastrelax_cst" scorefxn="scorefxn_full" task_operations="init,ex1ex2,repack_selected_res,dont_repack" repeats="1" cst_file="{1}" />
    <FastRelax name="fastrelax_nocst" scorefxn="scorefxn_nocst" task_operations="init,ex1ex2,repack_selected_res,dont_repack" repeats="1" />
  </MOVERS>

  <PROTOCOLS>
    <Add mover="add_ca_csts"/>
    <Add mover="add_lig_d_csts"/>
    <Add mover="fastrelax_cst"/>
    <Add mover="rm_csts"/>
    <Add mover="rm_lig_d_csts"/>
    <Add filter="ddg_after_relax_cst"/>
    <Add mover="fastrelax_nocst"/>

    <Add filter="ddg"/>
    <Add filter="cms"/>
    <Add filter="res_totalscore"/>
    <Add filter="totalscore"/>
  </PROTOCOLS>

</ROSETTASCRIPTS>

"""

XML_BSITE_REPACK_MIN_BETA_LIG_FIX = """
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="scorefxn_full" weights="beta_genpot">
      <Reweight scoretype="coordinate_constraint" weight="0.5"/>
      <Reweight scoretype="atom_pair_constraint" weight="0.3" />
    </ScoreFunction>
    <ScoreFunction name="scorefxn_soft" weights="beta_genpot_soft">
      <Reweight scoretype="coordinate_constraint" weight="1.0"/>
      <Reweight scoretype="fa_rep" weight="0.2"/>
      <Reweight scoretype="atom_pair_constraint" weight="0.5" />
    </ScoreFunction>
  </SCOREFXNS>

  <RESIDUE_SELECTORS>
    <Index name="repack_res" resnums="{0}"/>
    <Not name="fix_res" selector="repack_res"/>
    <Chain name="chB" chains="B,C"/>
  </RESIDUE_SELECTORS>

  <TASKOPERATIONS>
    <OperateOnResidueSubset name="dont_repack" selector="fix_res">
      <PreventRepackingRLT/>
    </OperateOnResidueSubset>
    <OperateOnResidueSubset name="repack_selected_res" selector="repack_res">
      <RestrictToRepackingRLT/>
    </OperateOnResidueSubset>
    #
    <InitializeFromCommandline name="init"/>
    <ExtraRotamersGeneric name="ex1ex2" ex1="1" ex2="1"/>
  </TASKOPERATIONS>

  <FILTERS>
    <ScoreType name="totalscore" scorefxn="scorefxn_full" threshold="9999999" confidence="1"/>
    <ScoreType name="fa_rep" score_type="fa_rep" scorefxn="scorefxn_full" threshold="9999999" confidence="1"/>

    <ResidueCount name="nres" confidence="1" />
    <CalculatorFilter name="res_totalscore" confidence="1" equation="SCORE/NRES" threshold="9999999">
      <Var name="SCORE" filter_name="totalscore" />
      <Var name="NRES" filter_name="nres" />
    </CalculatorFilter>


    <Ddg name="ddg" threshold="9999999" jump="1" repeats="1" repack="1" repack_bound="true" repack_unbound="true" relax_unbound="false" scorefxn="scorefxn_full"/>
    ContactMolecularSurface name="cms" target_selector="chB" binder_selector="repack_res"/>
    ContactMolecularSurface name="cms" target_selector="ligres" binder_selector="repack_res" use_rosetta_radii="true"/>

  </FILTERS>

  <MOVERS>
    <AddConstraints name="add_ca_csts" >
      <CoordinateConstraintGenerator name="coord_cst_gen" ca_only="true" native="true" align_reference="true"/>
    </AddConstraints>
    <ConstraintSetMover name="add_lig_d_csts" add_constraints="1" cst_file="{1}" />
    <ConstraintSetMover name="rm_lig_d_csts" cst_file="none" />
    <RemoveConstraints name="rm_csts" constraint_generators="coord_cst_gen"/>
    <PackRotamersMover name="repack" scorefxn="scorefxn_full" task_operations="init,ex1ex2,repack_selected_res,dont_repack"/>
    <PackRotamersMover name="repack_soft" scorefxn="scorefxn_soft" task_operations="init,ex1ex2,repack_selected_res,dont_repack"/>
     MinMover name="min_full" bb="1" chi="1" jump="0" scorefxn="scorefxn_full"/>
    <MinMover name="min_full" bb="1" chi="1" jump="ALL" scorefxn="scorefxn_full"/>
    <MinMover name="min_soft" bb="1" chi="1" jump="ALL" scorefxn="scorefxn_soft"/>
  </MOVERS>

  <PROTOCOLS>
    <Add mover="add_ca_csts"/>
    <Add mover="add_lig_d_csts"/>
    <Add mover="repack_soft"/>
    <Add mover="min_soft"/>
    <Add mover="repack"/>
    Add mover="min_full"/>
    <Add mover="rm_csts"/>
    <Add mover="rm_lig_d_csts"/>
    <Add filter="fa_rep"/>
    <Add filter="ddg"/>
    Add filter="cms"/>
    <Add filter="res_totalscore"/>
    <Add filter="totalscore"/>
  </PROTOCOLS>

</ROSETTASCRIPTS>

"""