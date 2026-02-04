"""MCP Server implementation."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models import ToolExecuteRequest, ToolExecuteResponse, HealthResponse
from tools import get_tool, list_tools

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting MCP Server...")
    logger.info(f"Available tools: {len(list_tools())}")
    yield
    logger.info("Shutting down MCP Server...")


# Initialize FastAPI app
app = FastAPI(
    title="MCP Server",
    description="Model Context Protocol server for recommendation system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@app.post("/tools/list")
async def list_available_tools():
    """List all available tools."""
    tools = list_tools()
    return {"tools": tools, "count": len(tools)}


@app.post("/tools/execute", response_model=ToolExecuteResponse)
async def execute_tool(request: ToolExecuteRequest):
    """Execute a specific tool."""
    logger.info(f"Executing tool: {request.tool_name} with params: {request.parameters}")

    # Get the tool
    tool = get_tool(request.tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{request.tool_name}' not found")

    try:
        # Execute the tool
        result = await tool(**request.parameters)
        return ToolExecuteResponse(success=True, result=result)
    except TypeError as e:
        logger.error(f"Invalid parameters for tool {request.tool_name}: {e}")
        return ToolExecuteResponse(
            success=False,
            error=f"Invalid parameters: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error executing tool {request.tool_name}: {e}")
        return ToolExecuteResponse(
            success=False,
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app", host=settings.mcp_host, port=settings.mcp_port, reload=True
    )
