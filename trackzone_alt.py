import cv2
import pandas as pd
import numpy as np
import os
from ultralytics import solutions
from datetime import datetime
import torch


timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
def track_people_in_separate_zones(
    video_path, 
    output_csv,
    region_0=None,  # Will default to full frame
    region_1=[(791, 58), (763, 518), (1132, 513), (1122, 73)],
    confidence=0.5,
    iou=0.7,
    show_output=True
):
    # Check CUDA status
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU - processing will be slower")

    # First check if output CSV exists and remove it
    if os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"Removed existing CSV file: {output_csv}")

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open {video_path}"

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video loaded: {w}x{h} at {fps} FPS")

    # Define region_0 as full frame if not specified
    if region_0 is None:
        region_0 = [(0, 0), (w, 0), (w, h), (0, h)]

    # Initialize separate tracking data dictionaries for each region
    region0_tracking = {}  # {id: timestamp}
    region1_tracking = {}  # {id: timestamp}

    # Initialize trackers for both regions
    region0_tracker = solutions.TrackZone(
        show=False,
        region=region_0,
        model="yolo12l.pt",
        classes=[0],  # 0 = person in COCO dataset
        conf=confidence,
        iou=iou,
        #tracker="bytetrack.yaml",
        tracker="bytetrack.yaml",
        verbose=False,
        device=device
    )

    region1_tracker = solutions.TrackZone(
        show=False,  # We'll create a combined visualization
        region=region_1,
        model="yolo12l.pt",
        classes=[0],
        conf=confidence, #conf 0.8 and iou 0.6 originally
        iou=iou,
        #tracker="bytetrack.yaml",
        tracker="bytetrack.yaml",
        verbose=False,
        device=device
    )

    # Create video writer for output
    output_video_path = os.path.splitext(video_path)[0] + "_separate_zonesAlt.mp4"
    video_writer = cv2.VideoWriter(
        output_video_path, 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        fps, 
        (w, h)
    )
    print(f"Output video will be saved to {output_video_path}")

    # Process each frame
    frame_count = 0
    start_time = datetime.now()

    print("Beginning processing...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Reached end of video or failed to read frame.")
            break

        # Calculate timestamp in seconds
        timestamp = frame_count / fps

        # Create a copy of the frame for visualization
        display_frame = frame.copy()

        # Process frame with region 0 tracker
        region0_results = region0_tracker.process(frame.copy())

        # Process frame with region 1 tracker (independently)
        region1_results = region1_tracker.process(frame.copy())

        # Check for tracked objects in region 0
        if hasattr(region0_tracker, 'track_ids') and len(region0_tracker.track_ids) > 0:
            for track_id, cls in zip(region0_tracker.track_ids, region0_tracker.clss):
                if cls != 0:  # Skip if not a person
                    continue

                # Record new people in region 0
                if track_id not in region0_tracking:
                    region0_tracking[track_id] = timestamp
                    print(f"Frame {frame_count}: New person in Region 0, ID: {track_id}")

        # Check for tracked objects in region 1
        if hasattr(region1_tracker, 'track_ids') and len(region1_tracker.track_ids) > 0:
            for track_id, cls in zip(region1_tracker.track_ids, region1_tracker.clss):
                if cls != 0:  # Skip if not a person
                    continue

                # Record new people in region 1
                if track_id not in region1_tracking:
                    region1_tracking[track_id] = timestamp
                    print(f"Frame {frame_count}: New person in Bike Zone, ID: {track_id}")

        # Create visualization with both regions
        # Draw region boundaries
        cv2.polylines(display_frame, [np.array(region_0, np.int32)], True, (255, 0, 0), 2)
        cv2.polylines(display_frame, [np.array(region_1, np.int32)], True, (0, 255, 0), 2)

        # Add region labels
        cv2.putText(display_frame, "Total Capture Region", (region_0[0][0], region_0[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(display_frame, "Bike Zone", (region_1[0][0], region_1[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw stats on frame
        cv2.putText(display_frame, f"Total people: {len(region0_tracking)}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(display_frame, f"Bike Zone People: {len(region1_tracking)}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display current frame count
        cv2.putText(display_frame, f"Frame: {frame_count} | Time: {timestamp:.2f}s", (20, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Overlay bounding boxes from both trackers
        # Region 0 boxes (blue)
        for box, track_id in zip(region0_tracker.boxes, region0_tracker.track_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, f"R0:{track_id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Region 1 boxes (green)
        for box, track_id in zip(region1_tracker.boxes, region1_tracker.track_ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"R1:{track_id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame to output video
        video_writer.write(display_frame)

        # Display if requested
        if show_output:
            cv2.imshow("Separate Region Tracking", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Processing stopped by user.")
                break

        # Progress indicator (every 100 frames)
        if frame_count % 100 == 0 and frame_count > 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            fps_processing = frame_count / max(1, elapsed)
            print(f"Processed {frame_count} frames ({fps_processing:.1f} FPS)... " +
                  f"Total People: {len(region0_tracking)} people, Bike Zone: {len(region1_tracking)} people")

        frame_count += 1

    # Clean up
    elapsed_time = (datetime.now() - start_time).total_seconds()
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"\nProcessing complete!")
    print(f"Processed {frame_count} frames in {elapsed_time:.1f} seconds " + 
          f"({frame_count/elapsed_time:.1f} FPS)")
    # Create DataFrame with separate columns for each region
    # First, determine the maximum number of entries for padding
    max_entries = max(len(region0_tracking), len(region1_tracking))

    # Create lists for each column
    region0_ids = list(region0_tracking.keys())
    region0_timestamps = list(region0_tracking.values())
    region1_ids = list(region1_tracking.keys())
    region1_timestamps = list(region1_tracking.values())

    # Pad shorter lists with None
    region0_ids += [None] * (max_entries - len(region0_ids))
    region0_timestamps += [None] * (max_entries - len(region0_timestamps))
    region1_ids += [None] * (max_entries - len(region1_ids))
    region1_timestamps += [None] * (max_entries - len(region1_timestamps))

    # Create DataFrame
    df = pd.DataFrame({
        "ObjectIDregion0": region0_ids,
        "TimestampRegion0": region0_timestamps,
        "ObjectIDregion1": region1_ids,
        "TimestampRegion1": region1_timestamps
    })

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

    # Print summary statistics
    print(f"\n--- TRACKING RESULTS ---")
    print(f"Total people tracked: {len(region0_tracking)}")
    print(f"Bike Zone people tracked: {len(region1_tracking)}")

    percentage_in_bike_zone = (len(region1_tracking)/len(region0_tracking))*100 if len(region0_tracking) > 0 else 0
    summary_data = {
    "Metric": ["Total people", "Total in bike zone", "Percentage in bike zone"],
    "Value": [len(region0_tracking), len(region1_tracking), percentage_in_bike_zone]
}
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.splitext(output_csv)[0] + "_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary statistics saved to: {summary_csv}")

    return df

if __name__ == "__main__":
    # Define your regions and parameters. Use coordinate counter to get coords of the region.
    #region_0 = [(791, 58), (763, 518), (1132, 513), (1122, 73)] for the BEFORE
    #region_1 = [(1152, 334), (1379, 286), (1452, 633), (1130, 613)]

    region_0 = [(1143, 685), (862, 695), (863, 317), (1158, 311)]
    #[(984, 313), (1251, 318), (1265, 754), (961, 756)]
    #[(792, 188), (1244, 204), (1209, 818), (756, 814)]
    #region_1 = [(1600, 549), (1400, 566), (1400, 787), (1677, 812)]
    region_1 = [(1449, 554), (1652, 557), (1684, 939), (994, 870)]
    # = [(1449, 554), (1652, 557), (1684, 939), (994, 870)]
    #[(1649, 543), (1364, 578), (1290, 810), (1682, 841)]


    #region_1 = [(869, 280), (823, 706), (1171, 728), (1188, 313)] #This is potentially for 44
    video_file = "D:/TestFootage/After Video.MOV" #Input Path Here. Edit this.
    file_name = os.path.basename(video_file)  # Gets "the file path,img-7849_5kBUYct8.mov"
    file_name_no_ext = os.path.splitext(file_name)[0]  # Removes extension

    # Run the tracking function
    results = track_people_in_separate_zones(
        video_path=video_file,
        output_csv=f"D:/ExcelResults/{file_name_no_ext} separate_zone_tracking{timestamp}.csv", #OutputPath Here. Edit this.
        region_0=region_0,
        region_1=region_1,
        confidence=0.6, #previous value is 0.58, iou = 0.7
        iou=0.5,
        show_output=True
    )

    # Print some additional analysis if desired
    print("Analysis complete!")