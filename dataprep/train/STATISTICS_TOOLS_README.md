# Pipeline Progress & Statistics Tools

## Overview

파이프라인 실행 후 진행 상태를 추적하고 통계를 생성하는 도구들입니다.

## 생성되는 파일들

- **pipeline_progress.csv**: 각 complex의 stage별 상태 (success/failed/skip/pending)
- **pipeline_statistics.json**: 전체 통계 데이터 (JSON)
- **PIPELINE_SUMMARY_REPORT.txt**: 사람이 읽기 쉬운 종합 리포트

## 도구 사용법

### 1. Pipeline Progress 재생성

Status 디렉토리를 읽어서 `pipeline_progress.csv`를 생성/업데이트합니다.

```bash
python regenerate_progress.py --base_dir /path/to/base_dir
```

**출력 예시:**
```
INFO: Found 5460 complexes in status directory
INFO: ✓ Saved progress report to pipeline_progress.csv

PIPELINE PROGRESS SUMMARY
══════════════════════════════════════════════════

1_save_ligands:
  ✓ Success:  875 ( 16.0%)
  ✗ Failed:  2185 ( 40.0%)
  ⊘ Skipped:    0 (  0.0%)
  ⋯ Pending: 2400 ( 44.0%)

2_prep_target:
  ✓ Success: 2416 ( 44.2%)
  ✗ Failed:   644 ( 11.8%)
  ...
```

### 2. 통계 생성

Stage output CSV 파일들과 status 디렉토리를 분석하여 상세 통계를 생성합니다.

```bash
python pipeline_statistics.py --base_dir /path/to/base_dir
```

**생성 정보:**
- 단계별 생존율 (survival rate)
- 주요 실패 원인 분석
- 최종 데이터셋 통계:
  - 단백질 개수
  - Target/Off-target 개수
  - 생성된 pose 개수
  - 평균 pose per complex

### 3. 요약 리포트 생성

모든 통계를 종합하여 읽기 쉬운 텍스트 리포트를 생성합니다.

```bash
python generate_summary_report.py --base_dir /path/to/base_dir
```

**리포트 내용:**
1. Overall Pipeline Flow - 단계별 흐름과 생존율
2. Stage-by-Stage Status - 각 단계의 성공/실패 비율
3. Failure Analysis - 실패 원인 분석
4. Final Dataset Statistics - 최종 데이터셋 통계
5. Top Proteins - 상위 단백질 목록
6. Recommendations & Next Steps - 권장사항

## 실행 예시 (한번에 모두 실행)

```bash
# 현재 디렉토리에서 모든 분석 실행
BASE_DIR=/scratch/yunmin/data/graph/train/example/val

# 1. Progress 업데이트
python regenerate_progress.py --base_dir $BASE_DIR

# 2. 통계 생성
python pipeline_statistics.py --base_dir $BASE_DIR

# 3. 요약 리포트 생성
python generate_summary_report.py --base_dir $BASE_DIR

# 결과 확인
cat $BASE_DIR/PIPELINE_SUMMARY_REPORT.txt
```

## One-liner

```bash
BASE_DIR=/scratch/yunmin/data/graph/train/example/val && \
  python /home/yunmin/proj/LigandMPNN/dataprep/train/regenerate_progress.py --base_dir $BASE_DIR && \
  python /home/yunmin/proj/LigandMPNN/dataprep/train/pipeline_statistics.py --base_dir $BASE_DIR && \
  python /home/yunmin/proj/LigandMPNN/dataprep/train/generate_summary_report.py --base_dir $BASE_DIR
```

## 출력 파일 설명

### pipeline_progress.csv

각 complex의 모든 stage별 상태를 추적합니다.

```csv
complex_id,protein_key,target_het_id,offtarget_het_id,1_save_ligands,2_prep_target,...
complex_000000,O43353,0LI,,success,success,success,success,success,success
complex_000001,O43353,,,pending,pending,success,failed,success,pending
```

**Status 값:**
- `success`: 성공
- `failed`: 실패 (실패 원인은 `{stage}_error` 컬럼에 표시)
- `skip`: 건너뜀
- `pending`: 아직 실행되지 않음

### pipeline_statistics.json

```json
{
  "stage_counts": {
    "stage1": 231,
    "stage2": 231,
    "stage3": 215,
    "stage4": 173,
    "stage5": 173,
    "stage6": 173
  },
  "failure_reasons": {
    "1_save_ligands": {
      "Other Error": 2185
    },
    "4_vina_prep": {
      "Other Error": 2185,
      "PDBQT Conversion Error": 42
    }
  },
  "final_dataset": {
    "total_complexes": 173,
    "unique_proteins": 11,
    "unique_targets": 11,
    "unique_offtargets": 173,
    "total_poses": 4060,
    "avg_poses_per_config": 23.5
  }
}
```

## 실패 원인 분석

Status 디렉토리의 `.failed` 파일들을 읽어 실패 원인을 자동으로 분류합니다:

- **File Not Found**: 파일이 존재하지 않음
- **Timeout**: 시간 초과
- **Vina Execution Failed**: Vina 실행 실패
- **No Poses Generated**: Pose가 생성되지 않음
- **PDBQT Conversion Error**: PDBQT 변환 오류
- **Parsing Error**: 파싱 오류
- **Other Error**: 기타 오류

## 주의사항

1. `regenerate_progress.py`는 **status 디렉토리**를 기준으로 동작합니다
2. `pipeline_statistics.py`는 **stage*_output.csv** 파일들이 필요합니다
3. `generate_summary_report.py`는 위 두 스크립트를 먼저 실행해야 합니다

## Troubleshooting

### "Status directory not found"
- `--base_dir`에 `status/` 디렉토리가 있는지 확인

### "Stage CSV not found"
- `stage1_output.csv` ~ `stage6_output.csv` 파일들이 있는지 확인
- 없어도 status 디렉토리만으로 progress는 생성 가능 (메타데이터 없이)

### 통계가 이상함
- Stage CSV 파일과 status 디렉토리의 complex_id 형식이 일치하는지 확인
- `pipeline_progress.csv`를 먼저 재생성한 후 통계 생성
