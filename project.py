import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 0. 환경 설정 (한글 폰트 및 스타일)
# =========================================================
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.1f}'.format)

# =========================================================
# 1. 데이터 로드 및 전처리 (Preprocessing)
# =========================================================
file_path = '아파트(매매)_실거래가_20251207162315.csv'
# 국토부 파일은 보통 헤더(설명)가 15줄 정도 있어서 header=15 옵션이 필요합니다.
df = pd.read_csv(file_path, encoding='cp949', header=15)

# (1) 숫자 변환: "15,000" -> 15000
df['거래금액'] = df['거래금액(만원)'].str.replace(',', '').astype(int)

# (2) 파생 변수 생성
df['구'] = df['시군구'].apply(lambda x: x.split()[1]) # 구 이름 추출
df['평당가'] = df['거래금액'] / (df['전용면적(㎡)'] / 3.305785) # 평당가 계산
df['건축년대'] = (df['건축년도'] // 10) * 10 # 10년 단위 연식 그룹화 (예: 1990, 2000)

# =========================================================
# 2. 데이터 분석 (Data Analysis) - 수치 확인
# =========================================================
# [분석 1] 지역별 시장 현황 요약
district_stats = df.groupby('구').agg(
    평균거래금액=('거래금액', 'mean'),
    평균평당가=('평당가', 'mean'),
    거래량=('거래금액', 'count')
).sort_values(by='평균거래금액', ascending=False)

print("=== [1] 지역별 아파트 시장 현황 (비싼 순) ===")
print(district_stats)
print("\n")

# [분석 2] 평당가 기준 상위 10개 아파트
top_apartments = df.sort_values(by='평당가', ascending=False)[
    ['단지명', '구', '전용면적(㎡)', '건축년도', '거래금액', '평당가']
].head(10)

print("=== [2] 평당가 기준 TOP 10 아파트 ===")
print(top_apartments)
print("\n")

# =========================================================
# 3. 데이터 시각화 (Visualization) - 그래프 확인
# =========================================================

# (1) [Bar Chart] 구별 평균 거래금액 & 거래량 (subplot으로 한 번에 출력)
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# 왼쪽: 평균 거래금액
sns.barplot(x=district_stats.index, y=district_stats['평균거래금액'], palette='viridis', ax=ax[0])
ax[0].set_title('구별 평균 거래금액 (단위: 만원)')
ax[0].set_ylabel('금액')
ax[0].tick_params(axis='x', rotation=45)
ax[0].grid(axis='y', linestyle='--', alpha=0.7)

# 오른쪽: 거래량
sns.barplot(x=district_stats.index, y=district_stats['거래량'], palette='rocket', ax=ax[1])
ax[1].set_title('구별 거래량 (단위: 건)')
ax[1].set_ylabel('건수')
ax[1].tick_params(axis='x', rotation=45)
ax[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# (2) [Correlation Heatmap] 상관관계 분석
numeric_cols = ['거래금액', '전용면적(㎡)', '건축년도', '층', '평당가']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1)
plt.title('주요 변수 간 상관관계')
plt.show()

# (3) [Pivot Heatmap] 구별 x 건축년대별 평당가 (핵심 분석)
pivot_price = df.pivot_table(index='구', columns='건축년대', values='평당가', aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_price, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=0.5)
plt.title('구별 x 건축년대별 평균 평당가 (만원)')
plt.xlabel('건축년대')
plt.ylabel('구')
plt.show()
