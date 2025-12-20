import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from plugins.functions.web_search import web_search

async def test_search():
    print("Testing web_search...")
    try:
        # web_search is a sync function that returns ActionResponse
        result = web_search("today's news", engine="baidu")
        print(f"Status: {result.action}")
        print(f"Content length: {len(result.response) if result.response else 0}")
        if result.response:
            print(f"Snippet: {result.response[:200]}...")
    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    asyncio.run(test_search())
