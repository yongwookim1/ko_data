import json
import pandas as pd
from pathlib import Path

# 출력할 때 컬럼 너비 설정 (화면 잘림 방지)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 30) # 내용이 길면 30자에서 자름
pd.set_option('display.width', 200)

def load_and_compare():
    base_path = Path("results_topic")
    
    # 1. 파일 데이터 로드
    try:
        with open(base_path / "method1_single.json", "r", encoding="utf-8") as f:
            data_m1 = json.load(f)
        with open(base_path / "method2_multi.json", "r", encoding="utf-8") as f:
            data_m2 = json.load(f)
        with open(base_path / "method3_binary.json", "r", encoding="utf-8") as f:
            data_m3 = json.load(f)
    except FileNotFoundError:
        print("❌ 결과 파일이 없습니다. 먼저 분류 코드를 실행하세요.")
        return

    # 2. 데이터 정리 (Dictionary 형태로 변환)
    # Key: Topic, Value: 각 방법의 결과
    comparison_data = []

    # 주제 순서대로 정리
    for i in range(len(data_m1)):
        topic_full = data_m1[i]['topic']
        
        # 주제가 너무 길면 앞부분만 잘라서 보기 좋게 만듦
        topic_short = topic_full[:20] + "..." if len(topic_full) > 20 else topic_full

        # Method 1 결과 (String)
        res_m1 = data_m1[i]['result']

        # Method 2 결과 (List -> String)
        res_m2 = ", ".join(data_m2[i]['result'])

        # Method 3 결과 (List -> String)
        # matched_list가 비어있으면 (Safe인 경우) '-' 표시
        m3_list = data_m3[i].get('matched_list', [])
        res_m3 = ", ".join(m3_list) if m3_list else "-"

        comparison_data.append({
            "Topic (주제)": topic_short,
            "Method 1 (1개 강제)": res_m1,
            "Method 2 (다중 강제)": res_m2,
            "Method 3 (꼼꼼 검사)": res_m3
        })

    # 3. 데이터프레임 생성 및 출력
    df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*100)
    print(" 🧐 AI Risk Classification Comparison Table (분류 결과 비교표)")
    print("="*100)
    print(df.to_string(index=False)) # 인덱스 번호 없이 깔끔하게 출력
    print("="*100)

    # 4. CSV로 저장 (엑셀에서 열어보기 용)
    csv_path = "comparison_table.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig') # utf-8-sig 해야 엑셀에서 한글 안 깨짐
    print(f"\n✅ 엑셀용 CSV 파일로도 저장했습니다: {csv_path}")

if __name__ == "__main__":
    load_and_compare()