from pydantic import BaseModel, StrictBool


class ChatHistoryMigrationRequest(BaseModel):
    output_dir: str | None = None
    write_latest: StrictBool = True


class ChatHistoryMigrationResponse(BaseModel):
    status: str
    operation: str
    items_migrated: int
    output_file: str
    latest_file: str | None = None


class ChatHistoryIngestRequest(BaseModel):
    input_file: str | None = None
    dry_run: StrictBool = False


class ChatHistoryIngestResponse(BaseModel):
    status: str
    operation: str
    records_loaded: int
    records_upserted: int
    records_skipped: int
    collection: str
