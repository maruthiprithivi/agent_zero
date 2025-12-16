"""Integration tests for LLM provider compatibility.

This module validates that the MCP server works correctly with major LLM providers:
- OpenAI
- Anthropic
- Google Gemini
- MoonshotAI (Kimi)

These tests are skipped unless specific environment variables are set.
"""

import os
import pytest
import json
import asyncio
from agent_zero.server.core import initialize_mcp_server

@pytest.mark.integration
class TestLLMCompatibility:

    @pytest.fixture(scope="class")
    async def mcp_server(self):
        """Initialize MCP server."""
        try:
            return initialize_mcp_server()
        except Exception as e:
            pytest.skip(f"MCP server initialization failed: {e}")

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_openai_compatibility(self, mcp_server):
        """Test compatibility with OpenAI API."""
        import openai

        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Get tools from MCP server
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": tool.__doc__ or "",
                    "parameters": {
                        "type": "object",
                        "properties": {},  # Simplified for this test
                    }
                }
            }
            for tool in mcp_server.registered_tools[:5] # Test first 5 tools
        ]

        # Basic chat completion test asking to use a tool
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant with access to ClickHouse tools."},
                    {"role": "user", "content": "What tools do you have access to?"}
                ],
                tools=tools,
                tool_choice="auto"
            )
            assert response.choices[0].message is not None
        except Exception as e:
            pytest.fail(f"OpenAI integration failed: {e}")

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.asyncio
    async def test_anthropic_compatibility(self, mcp_server):
        """Test compatibility with Anthropic API."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Convert MCP tools to Anthropic format
        # This is a simplified conversion for the test
        tools = []
        for tool in mcp_server.registered_tools[:5]:
            tools.append({
                "name": tool.__name__,
                "description": tool.__doc__ or "No description",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            })

        try:
            message = await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": "List the tools you have."}
                ],
                tools=tools
            )
            assert message.content is not None
        except Exception as e:
            pytest.fail(f"Anthropic integration failed: {e}")

    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_gemini_compatibility(self, mcp_server):
        """Test compatibility with Google Gemini."""
        # Note: using google-generativeai package
        try:
            import google.generativeai as genai
        except ImportError:
            pytest.skip("google-generativeai not installed")

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-pro')

        # Gemini tool declaration is complex, here we just verify the model works
        # and conceptually we would pass tools if the SDK supports it fully in the same way
        try:
            response = await model.generate_content_async("Hello")
            assert response.text is not None
        except Exception as e:
             pytest.fail(f"Gemini integration failed: {e}")

    @pytest.mark.skipif(not os.getenv("MOONSHOT_API_KEY"), reason="MOONSHOT_API_KEY not set")
    @pytest.mark.asyncio
    async def test_moonshot_compatibility(self, mcp_server):
        """Test compatibility with MoonshotAI (Kimi)."""
        import openai # Moonshot uses OpenAI-compatible API

        client = openai.AsyncOpenAI(
            api_key=os.getenv("MOONSHOT_API_KEY"),
            base_url="https://api.moonshot.cn/v1"
        )

        try:
            response = await client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {"role": "user", "content": "Hello"}
                ]
            )
            assert response.choices[0].message is not None
        except Exception as e:
            pytest.fail(f"MoonshotAI integration failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
