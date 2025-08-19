# 🧱 처음부터 만드는 Deep Research 시스템

Deep Research는 최근 가장 인기 있는 AI 에이전트 애플리케이션 중 하나로 떠올랐습니다. [OpenAI](https://openai.com/index/introducing-deep-research/), [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system), [Perplexity](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research), [Google](https://gemini.google/overview/deep-research/?hl=en) 모두 [다양한 소스](https://www.anthropic.com/news/research)의 컨텍스트를 활용해 포괄적인 보고서를 생성하는 딥 리서치 제품을 출시했습니다. 또한 많은 [오픈](https://huggingface.co/blog/open-deep-research) [소스](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) 구현체들도 있습니다. 저희는 사용자가 자신만의 모델, 검색 도구, MCP 서버를 가져올 수 있는 간단하고 설정 가능한 [오픈 딥 리서처](https://github.com/langchain-ai/open_deep_research)를 구축했습니다. 이 저장소에서는 딥 리서처를 처음부터 만들어보겠습니다! 다음은 우리가 구축할 주요 구성 요소들의 지도입니다:

해당 저장소는 실제 한국어 프롬프트를 기반으로 튜토리얼을 따라할 수 있도록 구성 중에 있으며, 최종적으로는 Deep Researcher를 만들어 저장소의 내용을 보다 다듬는 것 까지를 목표로 하고 있습니다.

![overview](https://github.com/user-attachments/assets/b71727bd-0094-40c4-af5e-87cdb02123b4)

## 🚀 빠른 시작 가이드

### 사전 준비사항

- **Node.js와 npx** (노트북 3에서 MCP 서버용으로 필요):
```bash
# Node.js 설치 (npx 포함)
# macOS에서 Homebrew 사용:
brew install node

# Ubuntu/Debian에서:
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# 설치 확인:
node --version
npx --version
```

- Python 3.11 이상을 사용하고 있는지 확인하세요.
- LangGraph와의 최적 호환성을 위해 이 버전이 필요합니다.
```bash
python3 --version
```
- [uv](https://docs.astral.sh/uv/) 패키지 매니저
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 새로운 uv 버전을 사용하기 위해 PATH 업데이트
export PATH="/Users/$USER/.local/bin:$PATH"
```

### 설치 방법

1. 저장소 클론:
```bash
git clone https://github.com/langchain-ai/deep_research_from_scratch
cd deep_research_from_scratch
```

2. 패키지와 의존성 설치 (가상환경을 자동으로 생성하고 관리합니다):
```bash
uv sync
```

3. 프로젝트 루트에 API 키가 포함된 `.env` 파일 생성:
```bash
# .env 파일 생성
touch .env
```

`.env` 파일에 API 키 추가:
```env
# 외부 검색을 사용하는 연구 에이전트에 필수
TAVILY_API_KEY=your_tavily_api_key_here

# 모델 사용에 필수
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 선택사항: 평가 및 추적용
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=deep_research_from_scratch
```

4. uv를 사용해 노트북이나 코드 실행:
```bash
# Jupyter 노트북 직접 실행
uv run jupyter notebook

# 또는 가상환경을 활성화하는 방법 (선호하는 경우)
source .venv/bin/activate  # Windows에서는: .venv\Scripts\activate
jupyter notebook
```

## 배경 지식

연구는 개방형 작업입니다. 사용자 요청에 답하는 최선의 전략을 미리 쉽게 알 수는 없습니다. 요청에 따라 다른 연구 전략과 다양한 수준의 검색 깊이가 필요할 수 있습니다.

[에이전트](https://langchain-ai.github.io/langgraph/tutorials/workflows/#agent)는 중간 결과를 사용해 탐색을 안내하면서 다양한 전략을 유연하게 적용할 수 있기 때문에 연구에 적합합니다. 오픈 딥 리서치는 3단계 프로세스의 일부로 에이전트를 사용해 연구를 수행합니다:

1. **범위 설정** – 연구 범위 명확화
2. **연구** – 연구 수행
3. **작성** – 최종 보고서 작성

## 📝 구성

이 저장소에는 딥 리서치 시스템을 처음부터 구축하는 5개의 튜토리얼 노트북이 포함되어 있습니다:

### 📚 튜토리얼 노트북

#### 1. 사용자 명확화 및 브리프 생성 (`notebooks/1_scoping.ipynb`)
**목적**: 연구 범위를 명확히 하고 사용자 입력을 구조화된 연구 브리프로 변환

**핵심 개념**:
- **사용자 명확화**: 구조화된 출력을 사용해 사용자로부터 추가 컨텍스트가 필요한지 판단
- **브리프 생성**: 대화를 상세한 연구 질문으로 변환
- **LangGraph 명령**: 흐름 제어 및 상태 업데이트를 위한 Command 시스템 사용
- **구조화된 출력**: 신뢰할 수 있는 의사결정을 위한 Pydantic 스키마

**구현 하이라이트**:
- 2단계 워크플로: 명확화 → 브리프 생성
- 환각을 방지하는 구조화된 출력 모델 (`ClarifyWithUser`, `ResearchQuestion`)
- 명확화 필요성에 따른 조건부 라우팅
- 컨텍스트에 민감한 연구를 위한 날짜 인식 프롬프트

**배울 내용**: 상태 관리, 구조화된 출력 패턴, 조건부 라우팅

---

#### 2. 커스텀 도구를 사용한 연구 에이전트 (`notebooks/2_research_agent.ipynb`)
**목적**: 외부 검색 도구를 사용하는 반복적 연구 에이전트 구축

**핵심 개념**:
- **에이전트 아키텍처**: LLM 결정 노드 + 도구 실행 노드 패턴
- **순차적 도구 실행**: 신뢰할 수 있는 동기식 도구 실행
- **검색 통합**: 콘텐츠 요약이 포함된 Tavily 검색
- **도구 실행**: 도구 호출이 포함된 ReAct 스타일 에이전트 루프

**구현 하이라이트**:
- 신뢰성과 단순성을 위한 동기식 도구 실행
- 검색 결과 압축을 위한 콘텐츠 요약
- 조건부 라우팅이 포함된 반복적 연구 루프
- 포괄적인 연구를 위한 풍부한 프롬프트 엔지니어링

**배울 내용**: 에이전트 패턴, 도구 통합, 검색 최적화, 연구 워크플로 설계

---

#### 3. MCP를 사용한 연구 에이전트 (`notebooks/3_research_agent_mcp.ipynb`)
**목적**: Model Context Protocol (MCP) 서버를 연구 도구로 통합

**핵심 개념**:
- **Model Context Protocol**: AI 도구 접근을 위한 표준화된 프로토콜
- **MCP 아키텍처**: stdio/HTTP를 통한 클라이언트-서버 통신
- **LangChain MCP 어댑터**: MCP 서버를 LangChain 도구로 원활하게 통합
- **로컬 vs 원격 MCP**: 전송 메커니즘 이해

**구현 하이라이트**:
- MCP 서버 관리를 위한 `MultiServerMCPClient`
- 설정 기반 서버 설정 (파일시스템 예제)
- 도구 출력 표시를 위한 풍부한 포맷팅
- MCP 프로토콜에서 요구하는 비동기 도구 실행 (중첩된 이벤트 루프 불필요)

**배울 내용**: MCP 통합, 클라이언트-서버 아키텍처, 프로토콜 기반 도구 접근

---

#### 4. 연구 수퍼바이저 (`notebooks/4_research_supervisor.ipynb`)
**목적**: 복잡한 연구 작업을 위한 멀티 에이전트 조정

**핵심 개념**:
- **수퍼바이저 패턴**: 조정 에이전트 + 작업자 에이전트
- **병렬 연구**: 병렬 도구 호출을 사용한 독립적인 주제에 대한 동시 연구 에이전트
- **연구 위임**: 작업 할당을 위한 구조화된 도구
- **컨텍스트 격리**: 다른 연구 주제를 위한 별도의 컨텍스트 윈도우

**구현 하이라이트**:
- 2노드 수퍼바이저 패턴 (`supervisor` + `supervisor_tools`)
- 진정한 동시성을 위한 `asyncio.gather()` 사용 병렬 연구 실행
- 위임을 위한 구조화된 도구 (`ConductResearch`, `ResearchComplete`)
- 병렬 연구 지침이 포함된 향상된 프롬프트
- 연구 집계 패턴의 포괄적인 문서화

**배울 내용**: 멀티 에이전트 패턴, 병렬 처리, 연구 조정, 비동기 오케스트레이션

---

#### 5. 완전한 멀티 에이전트 연구 시스템 (`notebooks/5_full_agent.ipynb`)
**목적**: 모든 구성 요소를 통합하는 완전한 엔드투엔드 연구 시스템

**핵심 개념**:
- **3단계 아키텍처**: 범위 설정 → 연구 → 작성
- **시스템 통합**: 범위 설정, 멀티 에이전트 연구, 보고서 생성 결합
- **상태 관리**: 서브그래프 간 복잡한 상태 흐름
- **엔드투엔드 워크플로**: 사용자 입력부터 최종 연구 보고서까지

**구현 하이라이트**:
- 적절한 상태 전환을 통한 완전한 워크플로 통합
- 출력 스키마가 포함된 수퍼바이저 및 연구자 서브그래프
- 연구 종합을 통한 최종 보고서 생성
- 명확화를 위한 스레드 기반 대화 관리

**배울 내용**: 시스템 아키텍처, 서브그래프 구성, 엔드투엔드 워크플로

---

### 🎯 주요 학습 성과

- **구조화된 출력**: 신뢰할 수 있는 AI 의사결정을 위한 Pydantic 스키마 사용
- **비동기 오케스트레이션**: 병렬 조정을 위한 비동기 패턴 vs 동기식 단순성의 전략적 사용
- **에이전트 패턴**: ReAct 루프, 수퍼바이저 패턴, 멀티 에이전트 조정
- **검색 통합**: 외부 API, MCP 서버, 콘텐츠 처리
- **워크플로 설계**: 복잡한 다단계 프로세스를 위한 LangGraph 패턴
- **상태 관리**: 서브그래프와 노드 간 복잡한 상태 흐름
- **프로토콜 통합**: MCP 서버 및 도구 생태계

각 노트북은 이전 개념을 기반으로 구축되어, 지능적인 범위 설정과 조정된 실행으로 복잡하고 다면적인 연구 쿼리를 처리할 수 있는 프로덕션 준비 딥 리서치 시스템으로 완성됩니다.

## 🇰🇷 한국어 개발자를 위한 추가 가이드

### API 키 발급 방법

이 튜토리얼을 진행하기 위해서는 다음 API 키들이 필요합니다:

#### 1. Tavily API 키 (필수)
- [Tavily 웹사이트](https://tavily.com/)에서 회원가입
- 대시보드에서 API 키 생성
- 무료 플랜으로도 충분히 테스트 가능

#### 2. OpenAI API 키 (필수)
- [OpenAI Platform](https://platform.openai.com/)에서 계정 생성
- 결제 정보 등록 (사용량 기반 과금)
- API Keys 섹션에서 새 키 생성

#### 3. Anthropic API 키 (필수)
- [Anthropic Console](https://console.anthropic.com/)에서 계정 생성
- 크레딧 구매 또는 무료 크레딧 사용
- API 키 생성

#### 4. LangSmith API 키 (선택사항)
- [LangSmith](https://smith.langchain.com/)에서 계정 생성
- 무료 플랜으로 시작 가능
- 모델 성능 추적 및 디버깅에 유용

### 개발 환경 설정 팁

1. **Python 버전 확인**: `python3 --version`으로 3.11 이상인지 확인
2. **가상환경 사용**: uv가 자동으로 관리하지만, 필요시 수동 활성화 가능
3. **Jupyter 설정**: VS Code의 Jupyter 확장을 사용하면 더 편리
4. **API 키 보안**: `.env` 파일을 `.gitignore`에 추가하여 실수로 커밋하지 않도록 주의

### 학습 순서 추천

1. **1번 노트북부터 순서대로**: 각 노트북이 이전 내용을 기반으로 구축됨
2. **코드 실행하며 이해**: 단순히 읽기보다는 직접 실행해보며 학습
3. **에러 해결 경험**: 에러가 발생하면 디버깅 과정을 통해 더 깊이 이해
4. **커스터마이징**: 기본 예제를 자신만의 용도로 수정해보기

### 한국어 데이터 활용

- 한국어 검색 쿼리로 테스트해보기
- 한국 관련 주제로 연구 진행해보기
- 결과 보고서를 한국어로 생성하도록 프롬프트 수정

이 튜토리얼을 통해 최신 AI 에이전트 기술을 활용한 딥 리서치 시스템을 완전히 이해하고 구축할 수 있게 될 것입니다!