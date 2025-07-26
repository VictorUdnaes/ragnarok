from pydantic import BaseModel, Field

class QueriesFromPlan(BaseModel):
    queries: list[str] = Field(
        description="List of queries generated from the plan steps, should be in sorted order"
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.queries = sorted(self.queries)  # Ensure queries are sorted