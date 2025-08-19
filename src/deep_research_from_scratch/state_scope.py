
"""연구 스코핑을 위한 상태 정의 및 Pydantic 스키마.

연구 에이전트 스코핑 워크플로우에 사용되는 상태 객체와 구조화된 스키마를
정의합니다. 연구자 상태 관리 및 출력 스키마를 포함합니다.
"""

import operator
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ===== 상태 정의 =====

class AgentInputState(MessagesState):
    """전체 에이전트의 입력 상태 - 사용자 입력의 메시지만 포함."""
    pass

class AgentState(MessagesState):
    """
    전체 멀티 에이전트 연구 시스템의 메인 상태.

    연구 조정을 위한 추가 필드로 MessagesState를 확장합니다.
    참고: 일부 필드는 서브그래프와 메인 워크플로우 간의 적절한
    상태 관리를 위해 다른 상태 클래스에서 중복됩니다.
    """

    # 사용자 대화 기록에서 생성된 연구 브리프
    research_brief: Optional[str]
    # 조정을 위해 슈퍼바이저 에이전트와 교환된 메시지
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # 연구 단계에서 수집된 원시 미처리 연구 노트
    raw_notes: Annotated[list[str], operator.add] = []
    # 보고서 생성을 위해 준비된 처리되고 구조화된 노트
    notes: Annotated[list[str], operator.add] = []
    # 최종 형식화된 연구 보고서
    final_report: str

# ===== 구조화된 출력 스키마 =====

class ClarifyWithUser(BaseModel):
    """사용자 명확화 결정 및 질문을 위한 스키마."""

    need_clarification: bool = Field(
        description="사용자에게 명확화 질문을 해야 하는지 여부.",
    )
    question: str = Field(
        description="보고서 범위를 명확히 하기 위해 사용자에게 할 질문",
    )
    verification: str = Field(
        description="사용자가 필요한 정보를 제공한 후 연구를 시작할 것이라는 확인 메시지.",
    )

class ResearchQuestion(BaseModel):
    """구조화된 연구 브리프 생성을 위한 스키마."""

    research_brief: str = Field(
        description="연구를 안내하는 데 사용될 연구 질문.",
    )
