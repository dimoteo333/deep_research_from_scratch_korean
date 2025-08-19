
"""사용자 명확화 및 연구 브리프 생성.

이 모듈은 연구 워크플로우의 스코핑 단계를 구현합니다:
1. 사용자의 요청에 명확화가 필요한지 평가
2. 대화에서 상세한 연구 브리프 생성

워크플로우는 구조화된 출력을 사용하여 연구를 진행하기에
충분한 컨텍스트가 있는지에 대한 결정론적 결정을 내립니다.
"""

from datetime import datetime
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research_from_scratch.prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from deep_research_from_scratch.state_scope import AgentState, ClarifyWithUser, ResearchQuestion, AgentInputState

# ===== 유틸리티 함수 =====

def get_today_str() -> str:
    """현재 날짜를 사람이 읽기 쉬운 형식으로 가져옵니다."""
    return datetime.now().strftime("%a %b %-d, %Y")

# ===== 설정 =====

# 모델 초기화
model = init_chat_model(model="openai:gpt-4.1", temperature=0.0)

# ===== 워크플로우 노드 =====

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    사용자의 요청에 연구를 진행하기에 충분한 정보가 포함되어 있는지 결정합니다.

    구조화된 출력을 사용하여 결정론적 결정을 내리고 환각을 방지합니다.
    연구 브리프 생성 또는 명확화 질문으로 종료하도록 라우팅합니다.
    """
    # 구조화된 출력 모델 설정
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # 명확화 지침으로 모델 호출
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]), 
            date=get_today_str()
        ))
    ])

    # 명확화 필요성에 따른 라우팅
    if response.need_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )

def write_research_brief(state: AgentState):
    """
    대화 기록을 포괄적인 연구 브리프로 변환합니다.

    구조화된 출력을 사용하여 브리프가 필요한 형식을 따르고
    효과적인 연구에 필요한 모든 세부사항을 포함하도록 합니다.
    """
    # 구조화된 출력 모델 설정
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # 대화 기록에서 연구 브리프 생성
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # 생성된 연구 브리프로 상태 업데이트하고 슈퍼바이저에게 전달
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }

# ===== 그래프 구성 =====

# 스코핑 워크플로우 구축
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# 워크플로우 노드 추가
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

# 워크플로우 엣지 추가
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

# 워크플로우 컴파일
scope_research = deep_researcher_builder.compile()
