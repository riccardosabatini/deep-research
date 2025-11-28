import asyncio
import aiosqlite
import json
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

class MyModel(BaseModel):
    data: str

class JsonSerializer:
    def dumps(self, obj) -> bytes:
        def default(o):
            if isinstance(o, BaseModel):
                return o.model_dump()
            raise TypeError(f"Type {type(o)} not serializable")
        return json.dumps(obj, default=default).encode("utf-8")

    def loads(self, data: bytes):
        return json.loads(data.decode("utf-8"))

async def main():
    async with aiosqlite.connect(":memory:") as conn:
        saver = AsyncSqliteSaver(conn, serde=JsonSerializer())
        await saver.setup()
        
        config = {"configurable": {"thread_id": "1"}}
        checkpoint = {
            "v": 1,
            "ts": "2023-01-01T00:00:00Z",
            "channel_values": {"my_key": MyModel(data="hello")},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        
        print("Saving checkpoint...")
        try:
            await saver.aput(config, checkpoint, {}, {})
            print("Saved successfully.")
        except Exception as e:
            print(f"Save failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
