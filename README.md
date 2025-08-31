# TAR 정령 자동 계산기 — Render 배포용 (M10, 모바일 간단 보기 포함)

- 앱 파일: `app.py` (브라우저 탭 제목 = "TAR 정령 자동 계산기")
- 추가 기능: "모바일 간단 보기" 토글(핵심 컬럼만 표기), 모바일 CSS 튜닝, ETA/조합 카운터

## Render 배포
1. 이 폴더를 GitHub에 푸시
2. Render → New → Web Service → 저장소 선택
3. Build: `pip install -r requirements.txt`
4. Start: `python app.py`
