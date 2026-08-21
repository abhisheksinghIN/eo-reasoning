from tools.agent_tools import TOOL_FUNCTIONS, TOOL_SCHEMAS


def test_agent_tool_names():
    assert set(TOOL_FUNCTIONS) == {"search_sentinel2", "analyze_temporal_aoi"}
    schema_names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert schema_names == set(TOOL_FUNCTIONS)
