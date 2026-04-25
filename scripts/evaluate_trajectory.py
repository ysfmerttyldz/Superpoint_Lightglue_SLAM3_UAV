#!/usr/bin/env python3
"""
SLAM Trajectory Evaluation Script
Usage: python evaluate_trajectory.py <camera_trajectory.txt> <ground_truth.csv>

Calculates Sim(3) RMSE and visualizes trajectories
Author: SP_SLAM3 Evaluation Tool
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from pathlib import Path

try:
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.metrics import PoseRelation
    import evo.main_ape as main_ape
except ImportError:
    print("ERROR: evo library not found!")
    print("Install with: pip install evo --break-system-packages")
    sys.exit(1)


def load_camera_trajectory(filepath):
    """
    Load camera trajectory - supports multiple formats:
    1. TUM format: timestamp tx ty tz qx qy qz qw (8 columns)
    2. Simple format: frame_id tx ty tz (4 columns)
    """
    print(f"📁 Loading camera trajectory: {filepath}")
    data = np.loadtxt(filepath)
    
    num_cols = data.shape[1]
    
    if num_cols == 8:
        # TUM format: timestamp tx ty tz qx qy qz qw
        print(f"   ✓ Detected TUM format (8 columns)")
        timestamps = data[:, 0]
        positions = data[:, 1:4]
        quaternions = np.column_stack([data[:, 7], data[:, 4], data[:, 5], data[:, 6]])  # wxyz
        
    elif num_cols == 4:
        # Simple format: frame_id tx ty tz
        print(f"   ✓ Detected simple format (4 columns: frame_id x y z)")
        timestamps = data[:, 0]
        positions = data[:, 1:4]
        # Use identity quaternions (no rotation information)
        quaternions = np.tile([1, 0, 0, 0], (len(data), 1))  # wxyz
        
    else:
        print(f"ERROR: Unsupported format! Expected 4 or 8 columns, got {num_cols}")
        print("Supported formats:")
        print("  1. TUM format (8 cols): timestamp tx ty tz qx qy qz qw")
        print("  2. Simple format (4 cols): frame_id tx ty tz")
        sys.exit(1)
    
    print(f"   ✓ Loaded {len(data)} poses")
    print(f"   ✓ Frame range: {int(timestamps[0])} → {int(timestamps[-1])}")
    
    return timestamps, positions, quaternions


def load_ground_truth(filepath):
    """
    Load ground truth - supports multiple formats:
    1. CSV with header (frame_numbers, translation_x/y/z)
    2. TUM format: timestamp tx ty tz qx qy qz qw (8 columns)
    3. Simple format: frame_id tx ty tz (4 columns)
    """
    print(f"📁 Loading ground truth: {filepath}")
    
    # Try to detect format
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    
    # Check if CSV with header
    if 'frame_numbers' in first_line or 'translation' in first_line or ',' in first_line:
        # CSV format with header
        print(f"   ✓ Detected CSV format with header")
        df = pd.read_csv(filepath)
        
        if 'frame_numbers' in df.columns:
            df['frame_id'] = df['frame_numbers'].str.extract(r'(\d+)').astype(int)
        elif 'frame_id' in df.columns:
            pass
        else:
            df['frame_id'] = np.arange(len(df))
        
        frames = df['frame_id'].values
        positions = df[['translation_x', 'translation_y', 'translation_z']].values
        
    else:
        # Text format - detect number of columns
        data = np.loadtxt(filepath)
        num_cols = data.shape[1]
        
        if num_cols == 8:
            # TUM format: timestamp tx ty tz qx qy qz qw
            print(f"   ✓ Detected TUM format (8 columns)")
            frames = data[:, 0].astype(int)
            positions = data[:, 1:4]
            
        elif num_cols == 4:
            # Simple format: frame_id tx ty tz
            print(f"   ✓ Detected simple format (4 columns: frame_id x y z)")
            frames = data[:, 0].astype(int)
            positions = data[:, 1:4]
            
        else:
            print(f"ERROR: Unsupported format! Expected 4 or 8 columns, got {num_cols}")
            print("Supported formats:")
            print("  1. CSV with header")
            print("  2. TUM format (8 cols): timestamp tx ty tz qx qy qz qw")
            print("  3. Simple format (4 cols): frame_id tx ty tz")
            sys.exit(1)
    
    print(f"   ✓ Loaded {len(frames)} poses")
    print(f"   ✓ Frame range: {frames[0]} → {frames[-1]}")
    
    return frames, positions


def align_trajectories(cam_timestamps, cam_positions, cam_quaternions, 
                       gt_frames, gt_positions, ratio=None):
    """
    Find common frames and align trajectories
    
    Args:
        ratio: Frame mapping ratio. If provided, camera frame i maps to GT frame i*ratio
               Example: ratio=2 means cam_frame_10 → gt_frame_20
    """
    print("\n🔗 Aligning trajectories...")
    
    cam_frames = cam_timestamps.astype(int)
    
    if ratio is not None:
        print(f"   ✓ Using frame ratio: {ratio} (cam_frame_i → gt_frame_{ratio}*i)")
        
        # Map camera frames to GT frames using ratio
        mapped_gt_frames = (cam_frames * ratio).astype(int)
        
        # Find which mapped frames exist in GT
        valid_mask = np.isin(mapped_gt_frames, gt_frames)
        valid_cam_frames = cam_frames[valid_mask]
        valid_mapped_gt_frames = mapped_gt_frames[valid_mask]
        
        if len(valid_cam_frames) == 0:
            print(f"ERROR: No frames found with ratio {ratio}!")
            print(f"Camera frames: {cam_frames[0]} → {cam_frames[-1]}")
            print(f"Mapped GT frames: {mapped_gt_frames[0]} → {mapped_gt_frames[-1]}")
            print(f"GT frame range: {gt_frames[0]} → {gt_frames[-1]}")
            sys.exit(1)
        
        print(f"   ✓ Matched frames: {len(valid_cam_frames)}")
        print(f"   ✓ Camera: {valid_cam_frames[0]} → {valid_cam_frames[-1]}")
        print(f"   ✓ GT (mapped): {valid_mapped_gt_frames[0]} → {valid_mapped_gt_frames[-1]}")
        
        # Get indices
        cam_idx = np.where(valid_mask)[0]
        gt_idx = np.array([np.where(gt_frames == f)[0][0] for f in valid_mapped_gt_frames])
        
        common_frames = valid_cam_frames
        
    else:
        # Original behavior: direct frame matching
        common_frames = np.intersect1d(gt_frames, cam_frames)
        
        if len(common_frames) == 0:
            print("ERROR: No common frames found!")
            sys.exit(1)
        
        print(f"   ✓ Common frames: {len(common_frames)} (frame {common_frames[0]} → {common_frames[-1]})")
        
        # Align data
        gt_idx = np.array([np.where(gt_frames == f)[0][0] for f in common_frames])
        cam_idx = np.array([np.where(cam_frames == f)[0][0] for f in common_frames])
    
    gt_pos_aligned = gt_positions[gt_idx]
    cam_pos_aligned = cam_positions[cam_idx]
    cam_quat_aligned = cam_quaternions[cam_idx]
    
    timestamps = common_frames.astype(float)
    
    return timestamps, gt_pos_aligned, cam_pos_aligned, cam_quat_aligned, common_frames


def calculate_rmse(gt_positions, cam_positions, cam_quaternions, timestamps):
    """Calculate Sim(3) aligned RMSE using evo"""
    print("\n🔧 Calculating RMSE with Sim(3) alignment...")
    
    # Create trajectory objects
    traj_ref = PoseTrajectory3D(
        positions_xyz=gt_positions,
        orientations_quat_wxyz=np.tile([1, 0, 0, 0], (len(gt_positions), 1)),
        timestamps=timestamps
    )
    
    traj_est = PoseTrajectory3D(
        positions_xyz=cam_positions,
        orientations_quat_wxyz=cam_quaternions,
        timestamps=timestamps
    )
    
    # Sim(3) alignment
    result_sim3 = main_ape.ape(
        traj_ref=traj_ref,
        traj_est=traj_est,
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True
    )
    
    stats = result_sim3.stats
    errors = result_sim3.np_arrays['error_array']
    
    # Get aligned trajectory
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True)
    cam_pos_aligned = traj_est_aligned.positions_xyz
    
    # Component-wise errors
    errors_x = np.abs(cam_pos_aligned[:, 0] - gt_positions[:, 0])
    errors_y = np.abs(cam_pos_aligned[:, 1] - gt_positions[:, 1])
    errors_z = np.abs(cam_pos_aligned[:, 2] - gt_positions[:, 2])
    
    rmse_x = np.sqrt(np.mean(errors_x**2))
    rmse_y = np.sqrt(np.mean(errors_y**2))
    rmse_z = np.sqrt(np.mean(errors_z**2))
    
    # XY plane RMSE
    errors_xy = np.sqrt((cam_pos_aligned[:, 0] - gt_positions[:, 0])**2 + 
                        (cam_pos_aligned[:, 1] - gt_positions[:, 1])**2)
    rmse_xy = np.sqrt(np.mean(errors_xy**2))
    
    # GT trajectory length
    gt_distances = np.sqrt(np.sum(np.diff(gt_positions, axis=0)**2, axis=1))
    gt_length = np.sum(gt_distances)
    
    return {
        'stats': stats,
        'errors': errors,
        'errors_x': errors_x,
        'errors_y': errors_y,
        'errors_z': errors_z,
        'errors_xy': errors_xy,
        'rmse_3d': stats['rmse'],
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'rmse_z': rmse_z,
        'rmse_xy': rmse_xy,
        'gt_length': gt_length,
        'cam_pos_aligned': cam_pos_aligned
    }


def print_results(results, common_frames):
    """Print results to terminal"""
    stats = results['stats']
    gt_length = results['gt_length']
    
    print("\n" + "="*80)
    print(" "*25 + "SIM(3) ALIGNMENT RESULTS")
    print("="*80)
    
    print(f"\n📊 MAIN METRICS:")
    print(f"{'─'*80}")
    print(f"   3D RMSE:       {results['rmse_3d']:8.4f} m  ({100*results['rmse_3d']/gt_length:6.2f}%)")
    print(f"   XY RMSE:       {results['rmse_xy']:8.4f} m  ({100*results['rmse_xy']/gt_length:6.2f}%)")
    print(f"   Z  RMSE:       {results['rmse_z']:8.4f} m  ({100*results['rmse_z']/gt_length:6.2f}%)")
    
    print(f"\n📐 COMPONENT-WISE RMSE:")
    print(f"{'─'*80}")
    print(f"   X Axis:        {results['rmse_x']:8.4f} m")
    print(f"   Y Axis:        {results['rmse_y']:8.4f} m")
    print(f"   Z Axis:        {results['rmse_z']:8.4f} m")
    
    print(f"\n📈 STATISTICS:")
    print(f"{'─'*80}")
    print(f"   Mean Error:    {stats['mean']:8.4f} m")
    print(f"   Median Error:  {stats['median']:8.4f} m")
    print(f"   Std Dev:       {stats['std']:8.4f} m")
    print(f"   Min Error:     {stats['min']:8.4f} m")
    print(f"   Max Error:     {stats['max']:8.4f} m")
    
    print(f"\n📏 TRAJECTORY INFO:")
    print(f"{'─'*80}")
    print(f"   GT Length:     {gt_length:8.2f} m")
    print(f"   Frames:        {len(common_frames):,}")
    
    # Percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\n📊 ERROR DISTRIBUTION (Percentiles):")
    print(f"{'─'*80}")
    for p in percentiles:
        val = np.percentile(results['errors'], p)
        print(f"   {p:2d}th:          {val:8.4f} m")
    
    # Verification
    calculated_3d = np.sqrt(results['rmse_xy']**2 + results['rmse_z']**2)
    print(f"\n✓ Verification: √(XY² + Z²) = {calculated_3d:.4f} m")
    print(f"✓ Actual 3D RMSE:            = {results['rmse_3d']:.4f} m")
    
    print("\n" + "="*80)


def visualize_trajectories(gt_positions, cam_pos_aligned, results, common_frames, output_path):
    """Create comprehensive visualization"""
    print(f"\n🎨 Creating visualization...")
    
    errors_3d = results['errors']
    errors_xy = results['errors_xy']
    errors_x = results['errors_x']
    errors_y = results['errors_y']
    errors_z = results['errors_z']
    rmse_3d = results['rmse_3d']
    rmse_xy = results['rmse_xy']
    rmse_z = results['rmse_z']
    rmse_x = results['rmse_x']
    rmse_y = results['rmse_y']
    gt_length = results['gt_length']
    
    fig = plt.figure(figsize=(24, 14))
    
    # 1. 3D Trajectory
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    ax1.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
             'g-', linewidth=4, alpha=0.9, label='Ground Truth', zorder=2)
    ax1.plot(cam_pos_aligned[:, 0], cam_pos_aligned[:, 1], cam_pos_aligned[:, 2], 
             'r--', linewidth=3, alpha=0.8, label='Estimate (Sim3)', zorder=1)
    ax1.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], 
               c='green', s=350, marker='o', edgecolors='black', linewidths=3, zorder=10)
    ax1.scatter(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2], 
               c='red', s=350, marker='s', edgecolors='black', linewidths=3, zorder=10)
    ax1.set_xlabel('X (m)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Y (m)', fontweight='bold', fontsize=12)
    ax1.set_zlabel('Z (m)', fontweight='bold', fontsize=12)
    ax1.set_title(f'3D Trajectory\nRMSE: {rmse_3d:.2f}m ({100*rmse_3d/gt_length:.2f}%)', 
                 fontweight='bold', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.view_init(elev=20, azim=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. XY Top View
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.plot(gt_positions[:, 0], gt_positions[:, 1], 'g-', linewidth=4, alpha=0.9)
    ax2.plot(cam_pos_aligned[:, 0], cam_pos_aligned[:, 1], 'r--', linewidth=3, alpha=0.8)
    ax2.scatter(gt_positions[0, 0], gt_positions[0, 1], c='green', s=250, marker='o', 
               edgecolors='black', linewidths=3, zorder=10)
    ax2.scatter(gt_positions[-1, 0], gt_positions[-1, 1], c='red', s=250, marker='s', 
               edgecolors='black', linewidths=3, zorder=10)
    ax2.set_xlabel('X (m)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Y (m)', fontweight='bold', fontsize=12)
    ax2.set_title(f'Top View (XY)\nXY RMSE: {rmse_xy:.2f}m', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. 3D Error over time
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.plot(common_frames, errors_3d, 'b-', linewidth=2, alpha=0.9)
    ax3.axhline(y=np.mean(errors_3d), color='red', linestyle='--', linewidth=2.5, 
               label=f'Mean: {np.mean(errors_3d):.2f}m')
    ax3.axhline(y=rmse_3d, color='orange', linestyle='--', linewidth=2.5, 
               label=f'RMSE: {rmse_3d:.2f}m')
    ax3.fill_between(common_frames, 0, errors_3d, alpha=0.3, color='blue')
    ax3.set_xlabel('Frame', fontweight='bold', fontsize=12)
    ax3.set_ylabel('3D Error (m)', fontweight='bold', fontsize=12)
    ax3.set_title('3D Error Over Time', fontweight='bold', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.hist(errors_3d, bins=50, color='skyblue', edgecolor='black', alpha=0.8)
    ax4.axvline(x=np.mean(errors_3d), color='red', linestyle='--', linewidth=3, 
               label=f'Mean: {np.mean(errors_3d):.2f}m')
    ax4.axvline(x=np.median(errors_3d), color='green', linestyle='--', linewidth=3, 
               label=f'Median: {np.median(errors_3d):.2f}m')
    ax4.set_xlabel('Error (m)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax4.set_title('Error Distribution', fontweight='bold', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. XY Error over time
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.plot(common_frames, errors_xy, 'purple', linewidth=2, alpha=0.9)
    ax5.axhline(y=rmse_xy, color='orange', linestyle='--', linewidth=2.5, 
               label=f'RMSE: {rmse_xy:.2f}m')
    ax5.fill_between(common_frames, 0, errors_xy, alpha=0.3, color='purple')
    ax5.set_xlabel('Frame', fontweight='bold', fontsize=12)
    ax5.set_ylabel('XY Error (m)', fontweight='bold', fontsize=12)
    ax5.set_title('Horizontal (XY) Error', fontweight='bold', fontsize=14)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Component comparison
    ax6 = fig.add_subplot(2, 4, 6)
    components = ['X', 'Y', 'Z', 'XY', '3D']
    rmse_values = [rmse_x, rmse_y, rmse_z, rmse_xy, rmse_3d]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#95E1D3']
    bars = ax6.bar(components, rmse_values, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    ax6.set_ylabel('RMSE (m)', fontweight='bold', fontsize=12)
    ax6.set_title('Component-wise RMSE', fontweight='bold', fontsize=14)
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmse_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}m', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 7. X/Y/Z components over time
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.plot(common_frames, gt_positions[:, 0], 'g-', linewidth=2.5, alpha=0.9, label='GT X')
    ax7.plot(common_frames, cam_pos_aligned[:, 0], 'r--', linewidth=2, alpha=0.8, label='Est X')
    ax7.set_xlabel('Frame', fontweight='bold', fontsize=12)
    ax7.set_ylabel('X Position (m)', fontweight='bold', fontsize=12)
    ax7.set_title(f'X Component (RMSE: {rmse_x:.2f}m)', fontweight='bold', fontsize=14)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # 8. Statistics summary
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    stats_text = f"""
╔════════════════════════════════╗
║     SIM(3) RESULTS SUMMARY     ║
╚════════════════════════════════╝

3D RMSE:     {rmse_3d:8.4f} m
XY RMSE:     {rmse_xy:8.4f} m
Z  RMSE:     {rmse_z:8.4f} m

Mean:        {np.mean(errors_3d):8.4f} m
Median:      {np.median(errors_3d):8.4f} m
Std:         {np.std(errors_3d):8.4f} m

Components:
  X:         {rmse_x:8.4f} m
  Y:         {rmse_y:8.4f} m
  Z:         {rmse_z:8.4f} m

Relative:    {100*rmse_3d/gt_length:8.2f} %
GT Length:   {gt_length:8.2f} m
Frames:      {len(common_frames):,}
"""
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7,
                       edgecolor='black', linewidth=2))
    
    fig.suptitle(f'Trajectory Evaluation - Sim(3) Alignment\n'
                f'RMSE: {rmse_3d:.2f}m | XY: {rmse_xy:.2f}m | Z: {rmse_z:.2f}m | '
                f'Frames: {len(common_frames)}', 
                fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Visualization saved: {output_path}")


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python evaluate_trajectory.py <camera_trajectory> <ground_truth> [ratio]")
        print("\n📋 Supported Formats:")
        print("\nCamera Trajectory:")
        print("  1. TUM format (8 cols):    timestamp tx ty tz qx qy qz qw")
        print("  2. Simple format (4 cols): frame_id tx ty tz")
        print("\nGround Truth:")
        print("  1. CSV with header:        frame_numbers,translation_x,translation_y,translation_z")
        print("  2. TUM format (8 cols):    timestamp tx ty tz qx qy qz qw")
        print("  3. Simple format (4 cols): frame_id tx ty tz")
        print("\n🔢 Frame Ratio (Optional):")
        print("  If provided, maps camera frame i to GT frame i*ratio")
        print("  Example: ratio=2 means cam_frame_10 → gt_frame_20")
        print("\n💡 Examples:")
        print("  python evaluate_trajectory.py CameraTrajectory.txt GT_Translation.csv")
        print("  python evaluate_trajectory.py estimate.txt groundtruth.txt")
        print("  python evaluate_trajectory.py estimate.txt groundtruth.txt 2")
        print("  python evaluate_trajectory.py estimate.txt groundtruth.txt 0.5")
        sys.exit(1)
    
    cam_file = sys.argv[1]
    gt_file = sys.argv[2]
    ratio = None
    
    # Parse ratio if provided
    if len(sys.argv) == 4:
        try:
            ratio = float(sys.argv[3])
            print(f"\n🔢 Frame ratio specified: {ratio}")
        except ValueError:
            print(f"ERROR: Invalid ratio '{sys.argv[3]}'. Must be a number.")
            sys.exit(1)
    
    # Check files exist
    if not Path(cam_file).exists():
        print(f"ERROR: Camera trajectory file not found: {cam_file}")
        sys.exit(1)
    
    if not Path(gt_file).exists():
        print(f"ERROR: Ground truth file not found: {gt_file}")
        sys.exit(1)
    
    print("="*80)
    print(" "*20 + "TRAJECTORY EVALUATION SCRIPT")
    print("="*80)
    
    # Load data
    cam_timestamps, cam_positions, cam_quaternions = load_camera_trajectory(cam_file)
    gt_frames, gt_positions = load_ground_truth(gt_file)
    
    # Align trajectories
    timestamps, gt_pos_aligned, cam_pos_aligned, cam_quat_aligned, common_frames = \
        align_trajectories(cam_timestamps, cam_positions, cam_quaternions, 
                          gt_frames, gt_positions, ratio=ratio)
    
    # Calculate RMSE
    results = calculate_rmse(gt_pos_aligned, cam_pos_aligned, cam_quat_aligned, timestamps)
    
    # Print results
    print_results(results, common_frames)
    
    # Visualize
    output_file = "trajectory_evaluation.png"
    visualize_trajectories(gt_pos_aligned, results['cam_pos_aligned'], 
                          results, common_frames, output_file)
    
    print("\n✅ Evaluation complete!")
    print(f"📊 Results printed to terminal")
    print(f"🎨 Visualization saved to: {output_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
