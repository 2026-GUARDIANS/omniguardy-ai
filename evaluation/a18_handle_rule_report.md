# A18 Handle ROI 보정 검증

기준선: `9a17efa` (`baseline before A18`). A18 보정은 AUDIO에서만 실행되고 KEYPAD에는 적용되지 않는다.

## 규칙과 설정

Trial 6 기본 wire 예측이 A17 또는 A18이고 `VISION_HANDLE_RULE=1`일 때만, 기존 진단과 동일한 별도 MediaPipe Pose pass를 실행한다. 기존 `_pose_features()`는 landmark를 공개하지 않고 집계 feature만 반환하므로 이를 변경·재사용하지 않았다.

* Pose: `models/pose_landmarker_lite.task`, VIDEO mode, `num_poses=2`, 10프레임 간격, 영상별 새 `PoseLandmarker`; `frame/fps*1000` timestamp는 영상 안에서 단조 증가한다.
* 다인 Pose: 진단과 동일하게 첫 번째 Pose만 사용한다. 양 손목 중 ROI 안의 visible 손목을 우선하고, 없으면 첫 visible 손목을 쓴다.
* gate: ROI sample >=3, consecutive >=2, elbow-angle range >=8°, wrist path >=0.05.
* A17 양성: A17/A18 점수를 교환하고 A18 confidence를 사용한다. 이는 Rule-based 후처리 점수이며 순수 모델 확률이 아니다.

환경변수: `VISION_HANDLE_RULE=0`(기본), `VISION_HANDLE_ROI=x0,y0,x1,y1`, `VISION_HANDLE_MIN_ROI_SAMPLES=3`, `VISION_HANDLE_MIN_CONSECUTIVE_SAMPLES=2`, `VISION_HANDLE_MIN_ELBOW_ANGLE_DEGREES=8`, `VISION_HANDLE_MIN_WRIST_PATH_LENGTH=0.05`. ROI/threshold가 잘못되면 경고를 로그하고 규칙을 비활성화한다.

## 결과

* Unit test: 10/10 PASS (양성, dwell/angle/path 부족, wrist 없음, invalid config, A17→A18 score swap, A18/N1/A19/A21 불변).
* 서빙 Pose 대 오프라인 진단: 46/46 일치. A18_20 27/27, A17 0/7, N1 0/12. 결과는 `evaluation/a18_rule_validation_20261011/serving_vs_diagnostic.csv`.
* ON HTTP Final Test 36 clip: A18_20 6개 A17→A18, A17 7/N1 12/A19 6/A21 5는 불변. Accuracy 0.8056→0.9722, Macro F1 0.7024→0.9624. 동일 환경 설계 데이터이므로 일반화 성능이 아니다.
* OFF 기준선 HTTP: 36/36 argmax와 확률 일치, max error 0 (`evaluation/integration_http_20261011`).
* ON latency: 전체 HTTP cold 11.366s, warm 14.900s; A17/A18 rule 대상은 대체로 15.0s 수준. OFF 기존 HTTP 평균 26.644s는 별도 서버 실행의 시스템 변동을 포함하므로 절대 비교가 아닌 참고치다. N1/A19/A21은 추가 Pose를 실행하지 않는다.
* KEYPAD regression: 압축/디버그본 기준 HTTP 200, `behavior:null`, SILENT_SIGNAL frame 37, details 구조 불변. rear-approach는 **미검증, 현장 두 사람 촬영 필요**.

## 현장 ROI

`python tools/draw_handle_roi.py <video.mp4> --frame 0`로 마우스 선택 후 출력된 `VISION_HANDLE_ROI=...`를 서버 재시작 전 설정한다. **ROI는 손잡이만 포함하고 초인종·인터폰 위치는 제외한다.** 설정 뒤 손잡이 조작→A18 보정, 초인종→A17 유지, 정상 출입→N1 유지를 각 3회 이상 시험한다. 불안정/오탐이면 `VISION_HANDLE_RULE=0`으로 재시작해 Trial 6 단독으로 복귀한다.
