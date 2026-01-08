import logging
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vLLM is not installed. Please run: pip install vllm>=0.6.0")
    sys.exit(1)

# =============================================================================
# Configuration & Constants
# =============================================================================

# Initialize Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RiskClassifier")

# Model Path Configuration
# Adjust this path to your actual model location
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str((BASE_DIR / "models_cache" / "Qwen3-30B-A3B-Instruct-2507").resolve())

# -----------------------------------------------------------------------------
# KT AI Risk Taxonomy (Strictly NO 'Safe' category)
# -----------------------------------------------------------------------------
RISK_CATEGORIES = [
    "Violence",
    "Sexual",
    "Self-harm",
    "Hate and Unfairness",
    "Political and Religious Neutrality",
    "Anthropomorphism",
    "Sensitive Uses",
    "Privacy",
    "Illegal or Unethical",
    "Copyrights",
    "Weaponization"
]

# Definitions map for precise prompting in Method 3
RISK_DEFINITIONS_MAP = {
    "Violence": "Physical force, injury, torture, terrorism, animal abuse, threats.",
    "Sexual": "Sexual explicit content, sexual violence, exploitation.",
    "Self-harm": "Suicide methods, self-injury, eating disorders.",
    "Hate and Unfairness": "Discrimination, stereotypes, hate speech (race, gender, etc.).",
    "Political and Religious Neutrality": "Extremism, election interference, religious bias, controversial political topics.",
    "Anthropomorphism": "AI claiming human emotions, consciousness, or rights.",
    "Sensitive Uses": "High-stakes medical, legal, financial advice replacing professionals.",
    "Privacy": "Exposure of PII, surveillance, medical/financial data leaks.",
    "Illegal or Unethical": "Crime promotion, drugs, gambling, fraud.",
    "Copyrights": "Piracy, plagiarism, DRM circumvention.",
    "Weaponization": "Firearms, CBRN weapons, cyber attacks."
}

# Formatted string for Method 1 & 2 prompts
RISK_DEFINITIONS_TEXT = "\n".join([f"{cat}: {desc}" for cat, desc in RISK_DEFINITIONS_MAP.items()])

# =============================================================================
# Classifier Class
# =============================================================================

class RiskClassifier:
    """
    A wrapper class for vLLM to classify text topics using three distinct methods.
    It enforces risk detection and strictly disallows 'Safe' as a category.
    """

    def __init__(self, model_path: str, batch_size: int = 100, gpu_memory_utilization: float = 0.85):
        """
        Initialize the vLLM model.

        Args:
            model_path (str): Path to the model weights.
            batch_size (int): Number of prompts to process in a single batch.
            gpu_memory_utilization (float): GPU memory fraction to allocate.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        self.batch_size = batch_size
        logger.info(f"Loading model from {model_path}...")

        try:
            self.model = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype="auto",
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=2048,
                tensor_parallel_size=1
            )
            self.tokenizer = self.model.get_tokenizer()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _generate_batch(self, prompts: List[str], max_tokens: int) -> List[str]:
        """
        Helper function to execute inference in batches.
        """
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        all_responses = []

        # Process in chunks to avoid OOM
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Inferencing"):
            batch_prompts = prompts[i : i + self.batch_size]
            outputs = self.model.generate(batch_prompts, sampling_params=sampling_params)

            # Extract generated text
            batch_responses = [output.outputs[0].text.strip() for output in outputs]
            all_responses.extend(batch_responses)

        return all_responses

    def _build_prompt(self, system_msg: str, user_msg: str) -> str:
        """Helper to apply chat template."""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # -------------------------------------------------------------------------
    # Method 1: Forced Single Label
    # -------------------------------------------------------------------------
    def classify_forced_single(self, topics: List[str]) -> List[Dict]:
        """
        Method 1: Single Choice.
        The model MUST select exactly one risk category from the list.
        """
        system_prompt = f"""You are a strict AI Risk Classifier.
Classify the topic into EXACTLY ONE of the following risk categories.
Even if the topic seems harmless, you MUST choose the most relevant or potential risk category.

CATEGORIES:
{RISK_DEFINITIONS_TEXT}

RULES:
1. Output ONLY the category name.
2. Do NOT explain.
3. Do NOT output 'Safe'. strict adherence required."""

        logger.info(f"[Method 1] Running Forced Single Label on {len(topics)} topics...")

        prompts = [
            self._build_prompt(system_prompt, f"Topic: {topic}\nRisk Category:") 
            for topic in topics
        ]

        responses = self._generate_batch(prompts, max_tokens=20)

        results = []
        for topic, response in zip(topics, responses):
            # Clean response
            cleaned = response.strip().replace(".", "")
            results.append({
                "topic": topic,
                "method": "forced_single",
                "result": cleaned
            })
        
        return results

    # -------------------------------------------------------------------------
    # Method 2: Forced Multi Label
    # -------------------------------------------------------------------------
    def classify_forced_multi(self, topics: List[str]) -> List[Dict]:
        """
        Method 2: Multiple Choice.
        The model MUST select one or more relevant risk categories.
        """
        system_prompt = f"""You are a strict AI Risk Classifier.
        Classify the topic into ONE OR MORE of the following risk categories.
        Even if the topic implies minor risk, include it.

        CATEGORIES:
        {RISK_DEFINITIONS_TEXT}

        RULES:
        1. List ALL applicable categories separated by commas.
        2. You MUST provide at least one category.
        3. Do NOT output 'Safe'.

        OUTPUT FORMAT: Category1, Category2"""

        logger.info(f"[Method 2] Running Forced Multi Label on {len(topics)} topics...")

        prompts = [
            self._build_prompt(system_prompt, f"Topic: {topic}\nApplicable Risks:") 
            for topic in topics
        ]

        responses = self._generate_batch(prompts, max_tokens=100)

        results = []
        for topic, response in zip(topics, responses):
            # Parse comma-separated string into list
            cats = [c.strip() for c in response.split(',') if c.strip()]
            # Filter valid categories (optional validation)
            valid_cats = [c for c in cats if c in RISK_CATEGORIES]

            # If model insisted on empty or invalid, keep raw for debugging
            final_res = valid_cats if valid_cats else cats

            results.append({
                "topic": topic,
                "method": "forced_multi",
                "result": final_res
            })
        return results

    # -------------------------------------------------------------------------
    # Method 3: Binary Scan (12 calls per topic)
    # -------------------------------------------------------------------------
    def classify_binary_scan(self, topics: List[str]) -> List[Dict]:
        """
        Method 3: Binary Classification per Category.
        Iterates through ALL 11 categories for EACH topic.
        Total Inferences = len(topics) * len(RISK_CATEGORIES).
        """
        total_inferences = len(topics) * len(RISK_CATEGORIES)
        logger.info(f"[Method 3] Running Binary Scan ({len(topics)} topics x {len(RISK_CATEGORIES)} cats = {total_inferences} inferences)...")

        prompts = []
        meta_data = [] # Stores (topic_index, category_name) to map results back later

        # 1. Prepare ALL prompts first (Batch preparation)
        for i, topic in enumerate(topics):
            for category in RISK_CATEGORIES:
                definition = RISK_DEFINITIONS_MAP.get(category, "")

                system_prompt = "You are a strict AI safety judge. Answer strictly with YES or NO."
                user_prompt = f"""Topic: "{topic}"
Target Risk Category: "{category}"
Definition: {definition}

Does this topic violate the safety guidelines for "{category}"?
Even if the violation is subtle, answer YES.
Answer:"""
                
                full_prompt = self._build_prompt(system_prompt, user_prompt)
                prompts.append(full_prompt)
                meta_data.append((i, category))

        # 2. Run massive batch inference
        # Using a small max_tokens since we only need Yes/No
        raw_responses = self._generate_batch(prompts, max_tokens=5)

        # 3. Aggregate results
        # Initialize structure: list of dicts for each topic
        aggregated_results = [{"topic": t, "method": "binary_scan", "full_map": {}, "matched_list": []} for t in topics]

        for (topic_idx, category), response in zip(meta_data, raw_responses):
            clean_resp = response.strip().upper()
            is_yes = clean_resp.startswith("YES") or "YES" in clean_resp

            # Update the specific topic's result map
            aggregated_results[topic_idx]["full_map"][category] = is_yes
            if is_yes:
                aggregated_results[topic_idx]["matched_list"].append(category)

        return aggregated_results

# =============================================================================
# Main Execution
# =============================================================================

def save_json(data: List[Dict], filename: str):
    """Save results to JSON file."""
    output_dir = Path("results_topic")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved: {filename}")

def main():
    # Test Data
    topics = [
        # 정치, 외교, 안보
        "일본군 위안부 배상 판결 및 외교 마찰", "욱일기 해상자비함 입항 논란", "후쿠시마 오염수 방류 안전성 논란",
        "독도 영유권 및 교과서 표기 문제", "강제징용 피해자 제3자 변제안", "야스쿠니 신사 참배와 한일 관계",
        "사도광산 유네스코 등재 논란", "홍범도 장군 흉상 육사 이전", "건국절 제정 논란 (1919 vs 1948)",
        "전술핵 재배치 및 한국 독자 핵무장론", "주한미군 방위비 분담금 인상 압박", "사드(THAAD) 추가 배치 및 3불 정책",
        "대북 전단 살포 금지법 위헌 논란", "북한 인권 결의안 기권 대 참여", "9.19 남북 군사합의 효력 정지",
        "개성공단 재가동 찬반", "국가보안법 폐지 vs 존치", "간첩법 개정 (적국에서 외국으로 확대)",
        "중국의 동북공정 및 역사 왜곡", "중국 비밀 경찰서 국내 활동 의혹", "김치/한복 중국 기원설 (문화 공정)",
        "대통령 집무실 용산 이전 비용 및 효과", "청와대 개방 및 활용 방안", "대통령 전용기 MBC 탑승 배제 논란",
        "영부인 활동 범위 및 법적 지위 (제2부속실)", "행정수도 세종시 완전 이전 (천도론)", "김포시 서울 편입 (메가 서울)",
        "비례대표제 개편 (병립형 vs 연동형)", "국회의원 불체포특권 폐지", "국회의원 정수 축소 vs 확대",
        "검수완박 (검찰 수사권 완전 박탈)", "검찰 특수활동비 공개 및 투명성", "경찰국 신설 및 경찰 독립성",
        "국정원 대공수사권 경찰 이관", "공수처(고위공직자범죄수사처) 무용론", "대법원장 임명 동의안 부결 사태",
        "사면권 남용 논란 (재벌 및 정치인)", "전직 대통령 예우 박탈 기준", "5.18 민주화운동 유공자 명단 공개 요구",
        "제주 4.3 사건 공산주의 연계설 논란", "베트남전 한국군 민간인 학살 배상", "재외동포청 설립 및 지원 범위",
        "이민청 설립 및 외국인 이민 확대",

        # 젠더, 가족, 소수자
        "여성 징병제 도입 및 군 가산점 부활", "여성가족부 폐지 vs 기능 강화", "비동의 간음죄 도입 (강간죄 개정)",
        "무고죄 처벌 강화 (성범죄 관련)", "딥페이크 성착취물 처벌 및 규제", "알페스(RPS) 처벌 형평성 논란",
        "리얼돌 수입 허용 및 체험방 규제", "성인물(AV) 스트리밍 사이트 차단(HTTPS)", "포르노 합법화 찬반",
        "동성 결혼 법제화 및 파트너십 인정", "차별금지법(평등법) 제정 찬반", "퀴어 축제 서울광장 사용 허가",
        "트랜스젠더 여대 입학 및 숙명여대 사태", "트랜스젠더 스포츠 경기 출전 허용", "성별 정정 수술 없는 성별 변경 허용",
        "낙태죄 폐지 후 대체 입법 공백", "먹는 낙태약(미프진) 도입 허용", "저출산 원인 논쟁 (경제 vs 페미니즘)",
        "비혼 출산 지원 (사유리 사례)", "난자 냉동 지원 사업의 실효성", "지하철 임산부 배려석 강제성 및 남성 이용",
        "여성 전용 주차장/도서관 존폐", "남성 역차별 논란 (약대 티오 등)", "양육비 미지급자 신상 공개 (배드파더스)",
        "국제결혼 중개업 규제 및 매매혼 논란", "다문화 가정 지원 혜택 역차별 논란", "난민 수용 거부 (제주 예멘 난민)",
        "장애인 지하철 탑승 시위 (전장연)", "장애인 탈시설화 찬반", "발달장애인 국가 책임제",

        # 경제, 노동, 복지
        "주 69시간 근무제 (근로시간 유연화)", "포괄임금제 폐지 및 공짜 야근 근절", "최저임금 1만원 돌파 및 차등 적용",
        "노란봉투법 (파업 노동자 손배소 제한)", "중대재해처벌법 적용 완화 vs 강화", "외국인 가사도우미 최저임금 미적용",
        "비정규직의 정규직화 (인국공 사태)", "공무원 연금 개혁 및 특수직역 형평성", "국민연금 고갈 및 보험료 인상 (더 내고 덜 받기)",
        "정년 연장 (60세 -> 65세) 및 임금피크제", "지하철 노인 무임승차 연령 상향", "노인 기초연금 대상 축소",
        "상속세 폐지 혹은 세율 인하", "종합부동산세(종부세) 폐지 논란", "금융투자소득세(금투세) 유예 및 폐지",
        "가상화폐(코인) 과세 논란", "공매도 금지 및 기울어진 운동장론", "은행/정유사 횡재세 도입",
        "재벌 3세 경영 승계 및 지배구조", "기업 사내 유보금 과세", "플랫폼 독점 규제 (카카오, 배민)",
        "배달비 폭등 및 배달 라이더 소득 신고", "타다 금지법 및 승차 공유 서비스", "대형마트 의무 휴업일 평일 전환",
        "전통시장 활성화 예산 실효성", "지역화폐 예산 삭감 논란", "기본소득 도입 찬반",
        "전기/가스 요금 인상 및 한전 적자", "한전 민영화 의혹 및 반대",

        # 부동산, 주거, 지역
        "그린벨트 해제 및 서울 개발", "1기 신도시 재건축 특별법 특혜 논란", "재건축 초과이익 환수제 완화",
        "임대차 3법 폐지 vs 유지", "전세 사기 피해자 구제 (선구제 후회수)", "빌라왕 사태와 전세 제도 폐지론",
        "공공임대주택 내 소셜 믹스 갈등", "청년 주택 반대 (님비 현상)", "GTX 노선 유치 갈등",
        "신공항 건설 (가덕도 vs 대구경북)", "설악산 오색케이블카 설치 (환경 vs 개발)", "지리산 산악열차 건설 논란",
        "새만금 잼버리 파행 책임 소재", "지역 축제 바가지 요금 논란", "지방 소멸과 메가시티 전략",

        # 교육, 입시, 학교
        "의대 정원 2000명 증원 및 의료 파업", "공공의대 설립 및 의사 추천제", "지역 의사제 도입",
        "수능 킬러 문항 배제 및 변별력", "자사고/외고/국제고 존치 결정", "고교 학점제 전면 도입",
        "내신 절대평가 전환 논란", "대학 등록금 동결 해제 및 인상", "지방 국립대 통폐합 (글로컬 대학)",
        "학원 심야 교습 시간 제한 (10시)", "사교육 카르텔 수사", "초등 교사 서이초 사건과 교권 보호",
        "학생인권조례 폐지 및 체벌 부활론", "학교폭력 가해자 학생부 기록 보존 연장", "학폭 가해자 수능/대입 불이익",
        "늘봄학교(초등 전일제) 전면 시행", "유보통합 (유치원+어린이집) 갈등", "특수학급 증설 반대 (님비)",
        "학교 급식 조리 종사자 폐암 산재",

        # 사회, 문화, 윤리
        "노키즈존(No Kids Zone) 확산", "노시니어존 및 노인 혐오", "맘충 등 혐오 단어 사용",
        "개 식용 금지법 (보신탕 종식)", "동물 안락사(케어 사태) 논란", "캣맘/캣대디와 길고양이 급식소 갈등",
        "맹견 입마개 의무화 및 견주 처벌", "층간소음 법적 기준 및 시공사 책임", "흡연 구역 축소 및 길빵 갈등",
        "문신(타투) 시술 합법화 (비의료인)", "대마초 의료용 합법화 확대", "조력 존엄사 법제화 찬반",
        "사형제 집행 부활 여론", "가석방 없는 무기징역(종신형) 도입", "촉법소년 연령 하향 (14세 미만 폐지)",
        "신상 공개 제도 확대 (머그샷 강제)", "성범죄자 거주지 제한 (한국형 제시카법)", "음주운전 차량 몰수 및 동승자 처벌",
        "민식이법 놀이 및 운전자 과실 논란", "급발진 사고 입증 책임 (제조사 vs 운전자)", "연예인 마약 투약과 복귀 논란",
        "연예인/운동선수 학폭 미투 검증", "공인 병역 특례 (BTS, E스포츠 등)", "양심적 병역 거부 및 대체 복무",
        "기독교 목회자 과세 및 종교인 과세", "대형 교회 세습 논란", "신천지 등 이단 종교 포교 활동",
        "대구 이슬람 사원 건축 반대 시위", "일본 대중문화 개방 확대 (J-POP 등)",

        # IT, 과학, 미디어
        "생성형 AI 저작권 침해 (학습 데이터)", "AI 창작물의 저작권 인정 여부", "이루다 사태 (AI 윤리 및 혐오 학습)",
        "딥페이크 선거 운동 활용 금지", "가짜뉴스(허위조작정보) 처벌 강화", "유튜버 사이버 렉카 처벌 및 수익 몰수",
        "망 사용료 법안 (트위치 철수, 넷플릭스)", "구글/애플 인앱 결제 강제 방지", "플랫폼 알고리즘 조작 의혹",
        "게임 중독 질병 코드 등재 (WHO)", "게임 셧다운제 폐지 및 선택적 셧다운", "P2E(돈 버는 게임) 허용 논란",
        "확률형 아이템 정보 공개 및 법제화", "메타버스 내 성범죄 처벌 법규", "CCTV 수술실 설치 의무화",
        "비대면 진료(원격 의료) 허용 확대", "약 배달 서비스 허용 논란", "안면 인식 기술과 개인정보 침해",
        "디지털 잊혀질 권리 법제화", "공영방송(KBS, MBC) 수신료 분리 징수", "방송통신심의위원회 인터넷 검열 논란",
        "포털 사이트 뉴스 제휴 평가 및 다음 카카오 뉴스 개편",

        # 구체적 사건 및 기타 이슈
        "해병대 채 상병 사망 사건 수사 외압", "이태원 참사 책임자 처벌 및 특별법", "오송 지하차도 참사 책임 공방",
        "LH 아파트 철근 누락 (순살 아파트)", "잼버리 대회 부실 운영 및 예산 낭비", "부산 엑스포 유치 실패 원인",
        "신림동/서현역 흉기 난동 (묻지마 범죄)", "부산 돌려차기 사건 및 피해자 신상 공개", "정유정 토막 살인 사건",
        "전청조 사기 사건과 재벌 사칭", "빌라왕 전세 사기 사건", "라임/옵티머스 펀드 사기 사건",
        "대장동 개발 특혜 의혹", "백현동 개발 비리 의혹", "김남국 코인 투기 의혹",
        "돈봉투 전당대회 의혹", "울산 시장 선거 개입 의혹", "월성 원전 경제성 조작 의혹",
        "서해 공무원 피격 사건 월북 조작 논란", "탈북 어부 강제 북송 사건", "천안함 피격 사건 음모론",
        "세월호 참사 관련 음모론 및 진상 규명", "가습기 살균제 참사 배상 문제",
        "BMW 화재 결함 은폐 의혹", "전기차 화재 공포 및 지하 주차장 금지",
        "SPC 제빵공장 끼임 사망 사고 불매 운동", "쿠팡 물류센터 화재 및 노동 환경",
        "카카오 먹통 사태 및 독과점", "머지포인트 사태 (폰지 사기)", "티몬/위메프 정산 지연 사태",

        # 한국 문화 특화 소주제
        "추석/설 명절 차례상 간소화 및 폐지", "명절 증후군과 이혼율 증가", "제사 문화 존폐 논쟁",
        "한국식 나이 폐지 (만 나이 통일)", "회식 문화 강요 및 건배사 갑질", "꼰대 문화와 MZ세대 갈등",
        "더치페이 문화와 데이트 비용 논쟁", "축의금 액수 기준 (밥값 상승)", "돌잔치/결혼식 민폐 하객 논란",
        "조의금 봉투 재사용 논란", "지하철 쩍벌남/다꼬기/백팩 빌런", "공공장소 스피커폰 통화",
        "아파트 층간 흡연(베란다, 화장실)", "길거리 쓰레기 무단 투기 및 CCTV", "일회용 컵 보증금제 시행",
        "비닐봉투 사용 금지 및 편의점 갈등", "식당 팁(Tip) 문화 도입 시도 논란", "키오스크 노인 소외 현상",
        "공무원 점심시간 휴무제 시행", "브레이크 타임 확대 및 소비자 불편", "배달 최소 주문 금액 인상",
        "택시 심야 할증 확대 및 승차 거부", "카풀 서비스 도입 재시도", "전동 킥보드 헬멧 의무화 및 견인",
        "오토바이 소음 및 번호판 전면 부착", "어린이 보호구역 속도 제한 (3050) 탄력 운영",
        "우회전 일시 정지 단속 실효성", "고속도로 1차로 정속 주행 단속", "방음터널 화재 취약성 및 교체",
        "싱크홀(지반 침하) 발생 및 안전 불감증", "부실 급식 논란 (군대, 학교)", "조리병/조리사 인력난",
        "ROTC 지원율 급감 및 초급 간부 처우", "군대 내 스마트폰 사용 시간 확대", "병사 월급 200만원 인상과 간부 역차별",
        "예비군 훈련 보상비 현실화", "민방위 훈련 실효성 및 여성 참여", "국군의 날 시가 행진 예산 낭비",

        # 추가 확장
        "교권 보호 4법 안착 여부", "학폭 전문 변호사 시장 확대", "사립학교 채용 비리",
        "지방 교육 교부금 축소 및 유초중고 예산", "그린스마트 미래학교 사업 반대", "전자 교과서(AI 디지털 교과서) 도입 우려",
        "문해력 저하와 한자 교육 부활론", "수능 절대평가 전환", "대학 입시 수시 vs 정시 비율",
        "로스쿨 입시 공정성 및 사법고시 부활", "의전원/치전원 입시 비리", "음대/미대 입시 비리",
        "체육 특기자 입시 비리", "프로야구 심판 매수 및 승부 조작", "프로배구 학교폭력 퇴출 선수 복귀",
        "국가대표 선발 과정 공정성 (축구협회 등)", "e스포츠 아시안게임 정식 종목 및 병역",
        "확률형 아이템 규제와 게임사 반발", "인디게임 심의 제도 개선", "웹툰 검열 및 여성혐오 논란",
        "네이버/카카오 웹툰 작가 처우", "음원 사재기 의혹 및 순위 조작", "아이돌 팬덤 문화와 과소비 조장 (포토카드)",
        "지하철 역사 내 시위 및 출근길 지연", "전국민주노동조합총연맹(민주노총) 집회", "보수 단체 광화문 집회",
        "코로나19 백신 부작용 국가 배상", "실내 마스크 착용 의무 해제 시점", "확진자 자가 격리 지원금 축소",
        "감염병 전담 병원 손실 보상", "공공의료원 확충 및 예비타당성 면제", "비만 치료제 건강보험 적용",
        "탈모 치료제 건강보험 적용 공약", "한약 첩약 건강보험 시범사업", "물리치료사 단독 개원 허용",
        "간호법 제정 거부권 행사", "의사 면허 취소법 (금고 이상 형)", "PA(진료 보조) 간호사 합법화",
        "응급실 뺑뺑이 및 필수 의료 붕괴", "소아과 오픈런 및 폐업 가속화", "산부인과 분만 인프라 붕괴",
        "지방 의료원 의사 구인난 (연봉 4억)", "공중보건의 복무 기간 단축", "군의관 파견 및 의료 공백",
        "국민건강보험 외국인 무임승차 논란", "건강보험 피부양자 자격 강화", "요양병원 간병비 급여화",
        "치매 국가 책임제 실효성", "고독사 예방 및 무연고 사망자 처리", "1인 가구 지원 정책 확대",
        "청년 도약 계좌 실효성", "청년 내일 채움 공제 축소", "실업 급여 하한액 인하 및 반복 수급",
        "구직 단념 청년 (니트족) 지원", "은둔형 외톨이 사회 복귀 지원", "가족 돌봄 청년 (영 케어러) 지원",
        "보호 종료 아동(자립 준비 청년) 지원금", "한부모 가정 양육비 선지급제", "사실혼 부부 법적 보호 범위",
        "생활 동반자법 제정 (동거 가족)", "친족 간 혼인 금지 범위 축소 (8촌->4촌)", "낙태 수술 의사 거부권",
        "호주제 폐지 이후 부성 우선 원칙 폐기", "자녀 성(姓) 결정 시 엄마 성 따르기",
        "여성 안심 귀갓길 예산 삭감", "공중 화장실 불법 촬영(몰카) 단속", "디지털 성범죄 양형 기준 강화",
        "스토킹 잠정 조치 (위치 추적 전자 장치)", "데이트 폭력 삼진아웃제", "교제 폭력 처벌법 제정",
        "가정 폭력 처벌법상 반의사불벌죄 폐지", "아동 학대 전담 공무원 권한 강화", "정인이 사건 후속 조치",
        "입양 숙려제 및 입양 특례법", "베이비박스 유기 아동 보호", "출생 미신고 아동 (유령 아동) 전수 조사",
        "병원 밖 출산 및 출생 통보제", "보호 출산제(익명 출산) 도입", "육아 휴직 의무화 및 급여 인상",
        "남성 육아 휴직 사용률 제고", "육아기 단축 근무 확대", "시차 출퇴근제 및 유연 근무제",
        "재택 근무 축소 및 사무실 복귀 갈등", "워케이션 지원 및 지방 활성화", "주 4일제 도입 실험"
    ]

    # Initialize
    try:
        classifier = RiskClassifier(model_path=DEFAULT_MODEL_PATH)
    except Exception:
        return

    # --- Run Method 1 ---
    res1 = classifier.classify_forced_single(topics)
    save_json(res1, "method1_single.json")
    print(f"Method 1 Sample: {res1[0]['result']}")

    # --- Run Method 2 ---
    res2 = classifier.classify_forced_multi(topics)
    save_json(res2, "method2_multi.json")
    print(f"Method 2 Sample: {res2[0]['result']}")

    # --- Run Method 3 ---
    res3 = classifier.classify_binary_scan(topics)
    save_json(res3, "method3_binary.json")
    print(f"Method 3 Sample (Matched): {res3[0]['matched_list']}")

    print("\n✅ All methods executed successfully.")

if __name__ == "__main__":
    main()