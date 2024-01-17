import asynchro
from frame_by_frame_analysis import frame_by_frame_analysis
def analysis_thread1():
    # Perform frame-by-frame analysis for thread 1
    frame_by_frame_analysis()

def analysis_thread2():
    # Perform frame-by-frame analysis for thread 2
    frame_by_frame_analysis()

# Create an asynchro task for each analysis thread
task1 = asynchro.Task(analysis_thread1)
task2 = asynchro.Task(analysis_thread2)

# Run the tasks concurrently
asynchro.run([task1, task2])
