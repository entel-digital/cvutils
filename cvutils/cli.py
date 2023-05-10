"""Console script for cvutils."""

import click


# @click.command()
# def main():
#     """Main entrypoint."""
#     click.echo("cvutils")
#     click.echo("=" * len("cvutils"))
#     click.echo("Computer vision auxiliary rutines")

@click.group()
@click.option('--debug/--no-debug', default=False)
#@click.pass_context
def main(debug):
    # click.echo('Hello World!')
    if debug:
        click.echo(f"Debug mode is on")


def open_cap_validator(source):
    try:
        return int(source)
    except:
        return source


@main.command()
@click.option('--source', default=0, help='stream source', type=open_cap_validator)
def test(source):
    from cvutils.pipeline_task.capture_video import CaptureVideo
    from cvutils.pipeline_task.fps_calculator import FPSCalculator
    from cvutils.pipeline_task.display_video import DisplayVideo
    from cvutils.pipeline_task.anotate_video import AnnotateVideo
    from cvutils.pipeline_task.aruco_finder import ArucoFinder
    from tqdm import tqdm

    # pipeline_task items
    capture_video = CaptureVideo(source)
    # click.echo("Capture created", err=True)
    # infer_landmarks = LandmarksRegresor(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    infer_aruco = ArucoFinder(source="image")
    # mode_manager = ModeManager(aruco_map, mode, debug=True)
    # analyse_coordinates = AnalysePose(['sentadillas'], store=True, store_path=f"{date_time_base_path}.csv")
    fps_calculator = FPSCalculator()
    annotate_video = AnnotateVideo("image", annotate_pose=False, annotate_fps=True, annotate_aruco=True)
    display_video = DisplayVideo("image", "Test 123", )
    # save_video = SaveVideo("image", f"{date_time_base_path}.avi")

    # Create image processing pipeline_task
    pipeline = (
                capture_video & infer_aruco & fps_calculator & annotate_video & display_video
    )
    # Iterate through pipeline_task
    try:
        # metrics.log_event('exercise start')
        for _ in tqdm(pipeline,
                      total=capture_video.frame_count if capture_video.frame_count > 0 else None,
                      disable=True):
            pass
    except StopIteration:
        return
    except KeyboardInterrupt:
        return
    finally:
        # Pipeline cleanup
        display_video.close()
        # analyse_coordinates.close()
        # save_video.close()
        # metrics.log_event('exercise end')


if __name__ == "__main__":
    main()  # pragma: no cover
