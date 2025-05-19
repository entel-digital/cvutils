import click
import time
from cvutils.pipeline_task.multi_capture_video import MultiCaptureVideo
from cvutils.pipeline_task.multi_display_video import MultiDisplayVideo
from cvutils.pipeline_task.fps_calculator import FPSCalculator
from cvutils.pipeline_task.anotate_video import AnnotateVideo
from cvutils.pipeline_task.yolo_detector import YOLODetector
from cvutils.pipeline_task.save_video import SaveVideo
from queue import Queue, Empty, Full

@click.group()
@click.option('--debug/--no-debug', default=False)
def main(debug):
    if debug:
        click.echo(f"Debug mode is on")

@main.command()
@click.argument('sources', nargs=-1, required=True)

def stream(sources):
    run_video(sources)

def run_video(sources):
    print(f"Received sources: {sources}")

    def convert_source(src):
        return int(src) if src.isdigit() else src

    converted_sources = [convert_source(src) for src in sources]
    capture_video = MultiCaptureVideo(converted_sources, target_fps=30)

    infer_yolo = YOLODetector(model_path='yolo11n.pt', device='cpu', verbose=False)

    # Define specific pipelines for each stream
    def create_pipeline_1():
        annotate_video = AnnotateVideo(dst="image", annotate_pose=False, annotate_fps=True)
        fps_calculator = FPSCalculator()
        return [infer_yolo, fps_calculator, annotate_video]

    def create_pipeline_2():
        fps_calculator = FPSCalculator()
        annotate_video = AnnotateVideo(dst="image", annotate_pose=False, annotate_fps=True)
        #save_video = SaveVideo(src="image", filename='output_pipeline_2.avi', fps=10, fourcc='MJPG')
        return [infer_yolo, fps_calculator, annotate_video]
    
    def create_pipeline_3():
        annotate_video = AnnotateVideo(dst="image", annotate_pose=False, annotate_fps=True)
        fps_calculator = FPSCalculator()
        return [infer_yolo, fps_calculator, annotate_video]

    def create_pipeline_4():
        fps_calculator = FPSCalculator()
        annotate_video = AnnotateVideo(dst="image", annotate_pose=False, annotate_fps=True)
        return [infer_yolo, fps_calculator, annotate_video]
        
    pipelines = [create_pipeline_1(), create_pipeline_2(), create_pipeline_3(), create_pipeline_4()]
  
    display_video = MultiDisplayVideo(src="image", window_name="Multi-Stream Wisely CV", window_size=(1840, 720))

    try:
        while True:
            try:
                combined_data = []
                data_list = next(capture_video.generator())
                for data in data_list:
                    source_index = data["source_index"]
                    pipeline = pipelines[source_index]
                    for step in pipeline:
                        data = step.map([data])[0]
                    combined_data.append(data)
                display_video.map(combined_data)
            except StopIteration:
                print("Capture completed")
                break
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        display_video.close()
        capture_video.cleanup()
        for pipeline in pipelines:
            for step in pipeline:
                if hasattr(step, 'close'):
                    step.close()

if __name__ == "__main__":
    main()