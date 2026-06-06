from crewai import Agent, Task, Crew
from crewai_tools import YoutubeVideoSearchTool

from crewai import LLM
from dotenv import load_dotenv
load_dotenv()
# Initialize the tool for general YouTube video searches
youtube_search_tool = YoutubeVideoSearchTool()

llm= LLM( model="gpt-4o-mini",
    temperature=0.3,)
video_researcher = Agent(
    role="Video Researcher",
    goal="Transcribe and analyze YouTube videos",
    backstory="Expert in extracting insights from video transcripts",
    memory = True,
    tools=[youtube_search_tool],
    verbose=True,
    llm=llm
)

research_task = Task(
    description="""
    Get the complete transcript of the YouTube video:
    {youtube_video_url}
    """,
    expected_output="""
    Full transcript of the video along with a summary of the content.
    """,
    agent=video_researcher,
)
# Create and run the crew
crew = Crew(agents=[video_researcher], tasks=[research_task])
result = crew.kickoff(inputs={"youtube_video_url": "https://youtu.be/BacJ6sEhqMo?si=yFsOcfgeIfJZmxH2"})
print(result)