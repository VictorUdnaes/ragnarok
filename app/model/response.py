from pydantic import BaseModel, field_validator, ValidationError

class RAGResponse(BaseModel):
    does_match: bool

    @field_validator('does_match')
    def validate_boolean(cls, value):
        if not isinstance(value, bool):
            raise ValueError('does_match must be a boolean value')
        return value