import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from ultralytics import YOLO
from patched_yolo_infer import auto_calculate_crop_values
import gc
from tqdm import tqdm
from scipy.spatial.distance import cdist
from collections import Counter
from sklearn.cluster import KMeans
import json
from datetime import datetime
import os
import pandas as pd

OUTPUT_BASE_DIR = r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\utility_scripts\optimal_patch_analysis_outputs"

def calculate_optimal_parameters_unified(results, combination_method, verbose=True):
    """
    Calculate optimal parameters treating each frame's recommendation as a complete configuration.
    """
    
    # Treat each frame as a complete configuration
    configurations = []
    for i in range(len(results['crop_x'])):
        config = {
            'crop_x': results['crop_x'][i],
            'crop_y': results['crop_y'][i], 
            'overlap_x': results['overlap_x'][i],
            'overlap_y': results['overlap_y'][i],
            'frame_idx': results['frame_indices'][i],
            'frame_time': results['frame_times'][i]
        }
        configurations.append(config)
    
    if verbose:
        print(f"\nAnalyzing {len(configurations)} complete configurations...")
        print("Individual frame recommendations:")
        for i, config in enumerate(configurations):
            print(f"  Frame {config['frame_idx']:3d}: ({config['crop_x']:4d}, {config['crop_y']:4d}, {config['overlap_x']:2d}%, {config['overlap_y']:2d}%)")
    
    # Apply different unified voting methods
    if combination_method == "exact_mode":
        optimal_config = exact_mode_voting(configurations, verbose)
    elif combination_method == "similarity_clustering":
        optimal_config = similarity_clustering_voting(configurations, verbose)
    elif combination_method == "centroid_distance":
        optimal_config = centroid_distance_voting(configurations, verbose)
    elif combination_method == "stability_weighted":
        optimal_config = stability_weighted_voting(configurations, verbose)
    elif combination_method == "conservative_config":
        optimal_config = conservative_config_voting(configurations, verbose)
    else:
        # Fallback to traditional method
        optimal_config = calculate_optimal_parameters(results, combination_method='median',verbose=verbose)
    
    return optimal_config

def exact_mode_voting(configurations, verbose=True):
    """Find the most frequently occurring exact configuration."""
    
    # Convert to tuples for hashing
    config_tuples = [(c['crop_x'], c['crop_y'], c['overlap_x'], c['overlap_y']) for c in configurations]
    
    # Count occurrences
    config_counts = Counter(config_tuples)
    
    if verbose:
        print(f"\nExact Mode Voting:")
        print(f"  Unique configurations: {len(config_counts)}")
        for config, count in config_counts.most_common():
            print(f"    {config}: {count} occurrence(s)")
    
    if len(config_counts.most_common()) > 0 and config_counts.most_common()[0][1] > 1:
        # Found repeated exact configuration
        most_common = config_counts.most_common()[0][0]
        optimal_config = {
            'CROP_WIDTH': most_common[0],
            'CROP_HEIGHT': most_common[1], 
            'OVERLAP_WIDTH_PERCENTAGE': most_common[2],
            'OVERLAP_HEIGHT_PERCENTAGE': most_common[3],
            'method': 'exact_mode',
            'confidence': config_counts.most_common()[0][1] / len(configurations)
        }
        if verbose:
            print(f"  → Selected most common: {most_common} (confidence: {optimal_config['confidence']:.2f})")
    else:
        # No exact repeats, fall back to similarity clustering
        if verbose:
            print(f"  → No exact repeats found, falling back to similarity clustering")
        optimal_config = similarity_clustering_voting(configurations, verbose)
        optimal_config['method'] = 'exact_mode_fallback'
    
    return optimal_config

def similarity_clustering_voting(configurations, verbose=True):
    """Group similar configurations and pick the centroid of the largest cluster."""
    
    # Convert to feature matrix (normalized for clustering)
    features = []
    for config in configurations:
        # Normalize features to similar scales
        features.append([
            config['crop_x'] / 1000,      # Normalize crop sizes to 0-2 range
            config['crop_y'] / 1000, 
            config['overlap_x'] / 100,    # Normalize overlaps to 0-1 range
            config['overlap_y'] / 100
        ])
    
    features = np.array(features)
    
    if len(configurations) < 3:
        # Too few points for clustering, use centroid
        optimal_config = centroid_distance_voting(configurations, verbose=False)
        optimal_config['method'] = 'similarity_clustering_fallback'
        return optimal_config
    
    # Determine optimal number of clusters (max 3, min 2)
    n_clusters = min(3, max(2, len(configurations) // 3))
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Find the largest cluster
    cluster_counts = Counter(cluster_labels)
    largest_cluster_id = cluster_counts.most_common()[0][0]
    largest_cluster_configs = [configurations[i] for i, label in enumerate(cluster_labels) if label == largest_cluster_id]
    
    if verbose:
        print(f"\nSimilarity Clustering Voting:")
        print(f"  Number of clusters: {n_clusters}")
        for cluster_id, count in cluster_counts.most_common():
            print(f"    Cluster {cluster_id}: {count} configurations")
        print(f"  → Selected largest cluster {largest_cluster_id} with {len(largest_cluster_configs)} configurations")
    
    # Calculate centroid of largest cluster
    centroid_config = {
        'CROP_WIDTH': int(np.round(np.mean([c['crop_x'] for c in largest_cluster_configs]))),
        'CROP_HEIGHT': int(np.round(np.mean([c['crop_y'] for c in largest_cluster_configs]))),
        'OVERLAP_WIDTH_PERCENTAGE': int(np.round(np.mean([c['overlap_x'] for c in largest_cluster_configs]))),
        'OVERLAP_HEIGHT_PERCENTAGE': int(np.round(np.mean([c['overlap_y'] for c in largest_cluster_configs]))),
        'method': 'similarity_clustering',
        'cluster_size': len(largest_cluster_configs),
        'confidence': len(largest_cluster_configs) / len(configurations)
    }
    
    if verbose:
        print(f"  → Centroid of largest cluster: ({centroid_config['CROP_WIDTH']}, {centroid_config['CROP_HEIGHT']}, {centroid_config['OVERLAP_WIDTH_PERCENTAGE']}%, {centroid_config['OVERLAP_HEIGHT_PERCENTAGE']}%)")
    
    return centroid_config

def centroid_distance_voting(configurations, verbose=True):
    """Find the configuration closest to the overall centroid."""
    
    # Calculate overall centroid
    centroid = {
        'crop_x': np.mean([c['crop_x'] for c in configurations]),
        'crop_y': np.mean([c['crop_y'] for c in configurations]),
        'overlap_x': np.mean([c['overlap_x'] for c in configurations]),
        'overlap_y': np.mean([c['overlap_y'] for c in configurations])
    }
    
    # Find configuration closest to centroid
    min_distance = float('inf')
    closest_config = None
    
    for config in configurations:
        # Calculate normalized Euclidean distance
        distance = np.sqrt(
            ((config['crop_x'] - centroid['crop_x']) / 1000) ** 2 +
            ((config['crop_y'] - centroid['crop_y']) / 1000) ** 2 +
            ((config['overlap_x'] - centroid['overlap_x']) / 100) ** 2 +
            ((config['overlap_y'] - centroid['overlap_y']) / 100) ** 2
        )
        
        if distance < min_distance:
            min_distance = distance
            closest_config = config
    
    optimal_config = {
        'CROP_WIDTH': closest_config['crop_x'],
        'CROP_HEIGHT': closest_config['crop_y'],
        'OVERLAP_WIDTH_PERCENTAGE': closest_config['overlap_x'],
        'OVERLAP_HEIGHT_PERCENTAGE': closest_config['overlap_y'],
        'method': 'centroid_distance',
        'source_frame': closest_config['frame_idx'],
        'distance_to_centroid': min_distance
    }
    
    if verbose:
        print(f"\nCentroid Distance Voting:")
        print(f"  Overall centroid: ({centroid['crop_x']:.1f}, {centroid['crop_y']:.1f}, {centroid['overlap_x']:.1f}%, {centroid['overlap_y']:.1f}%)")
        print(f"  → Selected frame {optimal_config['source_frame']} config: ({optimal_config['CROP_WIDTH']}, {optimal_config['CROP_HEIGHT']}, {optimal_config['OVERLAP_WIDTH_PERCENTAGE']}%, {optimal_config['OVERLAP_HEIGHT_PERCENTAGE']}%)")
        print(f"  → Distance to centroid: {min_distance:.3f}")
    
    return optimal_config

def stability_weighted_voting(configurations, verbose=True):
    """Weight configurations by how well they would work across all frames."""
    
    config_scores = []
    
    for candidate in configurations:
        # Calculate how "compatible" this config is with all other frames
        total_score = 0
        
        for test_config in configurations:
            # Score based on whether this config would be "reasonable" for each frame
            # Larger crops can handle smaller objects, higher overlap is generally safer
            
            crop_x_score = 1.0 if candidate['crop_x'] >= test_config['crop_x'] else candidate['crop_x'] / test_config['crop_x']
            crop_y_score = 1.0 if candidate['crop_y'] >= test_config['crop_y'] else candidate['crop_y'] / test_config['crop_y']
            overlap_x_score = 1.0 if candidate['overlap_x'] >= test_config['overlap_x'] else candidate['overlap_x'] / test_config['overlap_x']
            overlap_y_score = 1.0 if candidate['overlap_y'] >= test_config['overlap_y'] else candidate['overlap_y'] / test_config['overlap_y']
            
            frame_score = (crop_x_score + crop_y_score + overlap_x_score + overlap_y_score) / 4
            total_score += frame_score
        
        avg_score = total_score / len(configurations)
        config_scores.append((candidate, avg_score))
    
    # Sort by score and pick the best
    config_scores.sort(key=lambda x: x[1], reverse=True)
    best_config, best_score = config_scores[0]
    
    optimal_config = {
        'CROP_WIDTH': best_config['crop_x'],
        'CROP_HEIGHT': best_config['crop_y'],
        'OVERLAP_WIDTH_PERCENTAGE': best_config['overlap_x'],
        'OVERLAP_HEIGHT_PERCENTAGE': best_config['overlap_y'],
        'method': 'stability_weighted',
        'source_frame': best_config['frame_idx'],
        'stability_score': best_score
    }
    
    if verbose:
        print(f"\nStability Weighted Voting:")
        print(f"  Best configuration from frame {optimal_config['source_frame']}")
        print(f"  → Config: ({optimal_config['CROP_WIDTH']}, {optimal_config['CROP_HEIGHT']}, {optimal_config['OVERLAP_WIDTH_PERCENTAGE']}%, {optimal_config['OVERLAP_HEIGHT_PERCENTAGE']}%)")
        print(f"  → Stability score: {best_score:.3f}")
        
        if verbose and len(config_scores) <= 10:
            print("  All configurations ranked:")
            for i, (config, score) in enumerate(config_scores):
                print(f"    {i+1}. Frame {config['frame_idx']}: ({config['crop_x']}, {config['crop_y']}, {config['overlap_x']}%, {config['overlap_y']}%) → {score:.3f}")
    
    return optimal_config

def conservative_config_voting(configurations, verbose=True):
    """Select the configuration that provides the most coverage (largest crops, highest overlap)."""
    
    # Score each configuration by "coverage" (larger is more conservative)
    config_scores = []
    
    for config in configurations:
        # Coverage score: emphasize larger crops and higher overlaps
        coverage_score = (
            config['crop_x'] * config['crop_y'] / 1000000 +  # Crop area score
            (config['overlap_x'] + config['overlap_y']) / 200  # Overlap score
        )
        config_scores.append((config, coverage_score))
    
    # Sort by coverage score and pick the highest
    config_scores.sort(key=lambda x: x[1], reverse=True)
    best_config, best_score = config_scores[0]
    
    optimal_config = {
        'CROP_WIDTH': best_config['crop_x'],
        'CROP_HEIGHT': best_config['crop_y'],
        'OVERLAP_WIDTH_PERCENTAGE': best_config['overlap_x'],
        'OVERLAP_HEIGHT_PERCENTAGE': best_config['overlap_y'],
        'method': 'conservative_config',
        'source_frame': best_config['frame_idx'],
        'coverage_score': best_score
    }
    
    if verbose:
        print(f"\nConservative Configuration Voting:")
        print(f"  Selected highest coverage configuration from frame {optimal_config['source_frame']}")
        print(f"  → Config: ({optimal_config['CROP_WIDTH']}, {optimal_config['CROP_HEIGHT']}, {optimal_config['OVERLAP_WIDTH_PERCENTAGE']}%, {optimal_config['OVERLAP_HEIGHT_PERCENTAGE']}%)")
        print(f"  → Coverage score: {best_score:.3f}")
    
    return optimal_config

def analyze_video_for_optimal_crops_multi_method(
    video_path,
    num_samples=20,
    sampling_method="uniform",
    analysis_mode="network_based",
    model_path="yolo11m.pt",
    classes_list=None,
    conf_threshold=0.25,
    combination_methods=None,  # If None, will run all methods
    save_results=True,
    verbose=True
):
    """
    Analyze video frames using multiple combination methods and save each in separate folders.
    
    Args:
        video_path (str): Path to the video file
        num_samples (int): Number of frames to analyze
        sampling_method (str): How to select frames ("uniform", "random", "adaptive")
        analysis_mode (str): "network_based" or "resolution_based"
        model_path (str): Path to YOLO model for network_based analysis
        classes_list (list): List of class IDs to consider
        conf_threshold (float): Confidence threshold for detections
        combination_methods (list): List of combination methods to try. If None, uses all available methods.
        save_results (bool): Whether to save results to files
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Results from all combination methods
    """
    
    # Default combination methods if none specified
    if combination_methods is None:
        combination_methods = [
            "exact_mode",
            "similarity_clustering",
            "centroid_distance",
            "stability_weighted",
            "conservative_config",
            "individual_median"
        ]


    
    # Create main output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    main_output_dir = os.path.join(OUTPUT_BASE_DIR, f"{video_name}_multi_method_analysis_{timestamp}")
    
    if save_results:
        os.makedirs(main_output_dir, exist_ok=True)
        print(f"Results will be saved to: {main_output_dir}")
    
    # Open video and get properties (do this once for all methods)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create analysis info dictionary
    analysis_info = {
        "video_path": video_path,
        "video_name": video_name,
        "total_frames": total_frames,
        "fps": fps,
        "resolution": [width, height],
        "analysis_timestamp": timestamp,
        "num_samples": num_samples,
        "sampling_method": sampling_method,
        "analysis_mode": analysis_mode,
        "model_path": model_path,
        "classes_list": classes_list,
        "conf_threshold": conf_threshold,
        "combination_methods": combination_methods
    }
    
    print(f"Video Analysis: {video_path}")
    print(f"Properties: {total_frames} frames, {width}x{height}, {fps:.1f} FPS")
    print(f"Analyzing {num_samples} frames using {analysis_mode} mode...")
    print(f"Will test {len(combination_methods)} combination methods: {', '.join(combination_methods)}")
    
    # Load model if using network_based analysis (do this once)
    model = None
    if analysis_mode == "network_based":
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)
    
    # Determine frame indices to sample (do this once)
    if sampling_method == "uniform":
        frame_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
    elif sampling_method == "random":
        np.random.seed(42)  # For reproducibility
        frame_indices = np.random.choice(total_frames, num_samples, replace=False)
        frame_indices.sort()
    elif sampling_method == "adaptive":
        early = np.linspace(0, total_frames//4, num_samples//3, dtype=int)
        middle = np.linspace(total_frames//3, 2*total_frames//3, num_samples//3, dtype=int)
        late = np.linspace(3*total_frames//4, total_frames-1, num_samples//3, dtype=int)
        frame_indices = np.concatenate([early, middle, late])
        frame_indices = np.unique(frame_indices)
    else:
        frame_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
    
    # Collect raw results (do this once for all methods)
    results = {
        'crop_x': [], 
        'crop_y': [], 
        'overlap_x': [], 
        'overlap_y': [],
        'frame_indices': [], 
        'frame_times': []
    }
    
    if verbose:
        print(f"Sampling {len(frame_indices)} frames: {frame_indices}")
    
    # Analyze each sampled frame (do this once)
    successful_analyses = 0
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Analyzing frames")):
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                if verbose:
                    print(f"Could not read frame {frame_idx}, skipping...")
                continue
            
            crop_x, crop_y, overlap_x, overlap_y = auto_calculate_crop_values(
                image=frame,
                mode=analysis_mode,
                model=model,
                classes_list=classes_list,
                conf=conf_threshold
            )
            
            results['crop_x'].append(crop_x)
            results['crop_y'].append(crop_y)
            results['overlap_x'].append(overlap_x)
            results['overlap_y'].append(overlap_y)
            results['frame_indices'].append(frame_idx)
            results['frame_times'].append(frame_idx / fps)
            
            successful_analyses += 1
            
            if verbose:
                print(f"Frame {frame_idx:4d}: crop={crop_x:4d}x{crop_y:4d}, overlap={overlap_x:2d}%x{overlap_y:2d}%")
                
        except Exception as e:
            if verbose:
                print(f"Error analyzing frame {frame_idx}: {e}")
            continue
    
    cap.release()
    
    if len(results['crop_x']) == 0:
        raise ValueError("No frames could be analyzed successfully")
    
    if verbose:
        print(f"\nSuccessfully analyzed {successful_analyses} out of {len(frame_indices)} frames")
        print(f"\nTesting {len(combination_methods)} combination methods...")
    
    # Store results for each method
    all_method_results = {}
    method_comparison = []
    
    # Test each combination method
    for method in combination_methods:
        if verbose:
            print(f"\n--- Testing method: {method} ---")
        
        try:
            # Calculate optimal parameters for this method
            method_results = calculate_optimal_parameters_unified(results, method, verbose=False)
            
            # Add common info
            method_results['analysis_info'] = analysis_info.copy()
            method_results['analysis_info']['combination_method'] = method
            method_results['raw_results'] = results
            
            # Store results
            all_method_results[method] = method_results
            
            # Add to comparison
            method_comparison.append({
                'method': method,
                'crop_width': method_results['CROP_WIDTH'],
                'crop_height': method_results['CROP_HEIGHT'],
                'overlap_width': method_results['OVERLAP_WIDTH_PERCENTAGE'],
                'overlap_height': method_results['OVERLAP_HEIGHT_PERCENTAGE'],
                'confidence': method_results.get('confidence', 0),
                'stability_score': method_results.get('stability_score', 0),
                'source_frame': method_results.get('source_frame', 'N/A')
            })
            
            if verbose:
                print(f"  {method}: ({method_results['CROP_WIDTH']}, {method_results['CROP_HEIGHT']}, {method_results['OVERLAP_WIDTH_PERCENTAGE']}%, {method_results['OVERLAP_HEIGHT_PERCENTAGE']}%)")
            
            # Save results for this method in separate folder
            if save_results:
                method_output_dir = os.path.join(main_output_dir, f"method_{method}")
                os.makedirs(method_output_dir, exist_ok=True)
                save_analysis_results(method_results, method_output_dir, f"{video_name}_{method}")
                
        except Exception as e:
            if verbose:
                print(f"  ERROR with method {method}: {e}")
            continue
    
    # Save comparison summary
    if save_results and method_comparison:
        save_method_comparison(method_comparison, main_output_dir, video_name, analysis_info)
    
    # Print final comparison
    if verbose:
        print_method_comparison(method_comparison)
    
    # Cleanup
    if model is not None:
        del model
    gc.collect()
    
    return {
        'method_results': all_method_results,
        'comparison': method_comparison,
        'analysis_info': analysis_info,
        'raw_results': results
    }

def save_method_comparison(method_comparison, output_dir, video_name, analysis_info):
    """Save a comparison of all methods."""
    
    # Save as CSV
    comparison_df = pd.DataFrame(method_comparison)
    comparison_df.to_csv(os.path.join(output_dir, f"{video_name}_method_comparison.csv"), index=False)
    
    # Save as JSON
    comparison_data = {
        'analysis_info': analysis_info,
        'method_comparison': method_comparison,
        'summary': {
            'total_methods_tested': len(method_comparison),
            'most_conservative': max(method_comparison, key=lambda x: x['crop_width'] * x['crop_height']),
            'least_conservative': min(method_comparison, key=lambda x: x['crop_width'] * x['crop_height']),
            'highest_confidence': max(method_comparison, key=lambda x: x['confidence']),
            'highest_stability': max(method_comparison, key=lambda x: x['stability_score'])
        }
    }
    
    with open(os.path.join(output_dir, f"{video_name}_method_comparison.json"), 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    # Save detailed comparison report
    report_path = os.path.join(output_dir, f"{video_name}_method_comparison_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COMBINATION METHOD COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Video: {analysis_info['video_name']}\n")
        f.write(f"Analysis Date: {analysis_info['analysis_timestamp']}\n")
        f.write(f"Methods Tested: {len(method_comparison)}\n\n")
        
        f.write("RESULTS BY METHOD:\n")
        f.write("-"*80 + "\n")
        for result in method_comparison:
            f.write(f"\nMethod: {result['method'].upper()}\n")
            f.write(f"  Crop Size: {result['crop_width']} x {result['crop_height']}\n")
            f.write(f"  Overlap: {result['overlap_width']}% x {result['overlap_height']}%\n")
            f.write(f"  Confidence: {result['confidence']:.3f}\n")
            f.write(f"  Stability Score: {result['stability_score']:.1f}\n")
            f.write(f"  Source Frame: {result['source_frame']}\n")
        
        f.write(f"\nSUMMARY:\n")
        f.write("-"*80 + "\n")
        summary = comparison_data['summary']
        f.write(f"Most Conservative: {summary['most_conservative']['method']} ")
        f.write(f"({summary['most_conservative']['crop_width']}x{summary['most_conservative']['crop_height']})\n")
        f.write(f"Least Conservative: {summary['least_conservative']['method']} ")
        f.write(f"({summary['least_conservative']['crop_width']}x{summary['least_conservative']['crop_height']})\n")
        f.write(f"Highest Confidence: {summary['highest_confidence']['method']} ")
        f.write(f"({summary['highest_confidence']['confidence']:.3f})\n")
        f.write(f"Highest Stability: {summary['highest_stability']['method']} ")
        f.write(f"({summary['highest_stability']['stability_score']:.1f})\n")
    
    # Create comparison visualization
    create_method_comparison_plot(method_comparison, output_dir, video_name)

def create_method_comparison_plot(method_comparison, output_dir, video_name):
    """Create a visualization comparing all methods."""
    
    methods = [r['method'] for r in method_comparison]
    crop_widths = [r['crop_width'] for r in method_comparison]
    crop_heights = [r['crop_height'] for r in method_comparison]
    overlap_widths = [r['overlap_width'] for r in method_comparison]
    overlap_heights = [r['overlap_height'] for r in method_comparison]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Method Comparison: {video_name}', fontsize=16)
    
    # Crop widths
    axes[0, 0].bar(methods, crop_widths, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Crop Width by Method')
    axes[0, 0].set_ylabel('Width (pixels)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Crop heights  
    axes[0, 1].bar(methods, crop_heights, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Crop Height by Method')
    axes[0, 1].set_ylabel('Height (pixels)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Overlap widths
    axes[1, 0].bar(methods, overlap_widths, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Overlap Width % by Method')
    axes[1, 0].set_ylabel('Overlap %')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Overlap heights
    axes[1, 1].bar(methods, overlap_heights, color='gold', alpha=0.7)
    axes[1, 1].set_title('Overlap Height % by Method')
    axes[1, 1].set_ylabel('Overlap %')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{video_name}_method_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def print_method_comparison(method_comparison):
    """Print a formatted comparison of all methods."""
    
    print(f"\n" + "="*80)
    print("METHOD COMPARISON SUMMARY")
    print("="*80)
    
    # Create formatted table
    print(f"{'Method':<20} {'Crop W':<8} {'Crop H':<8} {'Ovlp W%':<8} {'Ovlp H%':<8} {'Conf':<8} {'Stab':<8}")
    print("-"*80)
    
    for result in method_comparison:
        print(f"{result['method']:<20} {result['crop_width']:<8} {result['crop_height']:<8} "
              f"{result['overlap_width']:<8} {result['overlap_height']:<8} "
              f"{result['confidence']:<8.3f} {result['stability_score']:<8.1f}")
    
    print("="*80)

def save_analysis_results(analysis_results, output_dir, video_name):
    """Save all analysis results to files."""
    
    # 1. Save optimal parameters as JSON
    optimal_params = {
        'CROP_WIDTH': analysis_results['CROP_WIDTH'],
        'CROP_HEIGHT': analysis_results['CROP_HEIGHT'],
        'OVERLAP_WIDTH_PERCENTAGE': analysis_results['OVERLAP_WIDTH_PERCENTAGE'],
        'OVERLAP_HEIGHT_PERCENTAGE': analysis_results['OVERLAP_HEIGHT_PERCENTAGE'],
        'method': analysis_results.get('method', 'unknown'),
        'confidence': analysis_results.get('confidence', 0),
        'analysis_info': analysis_results['analysis_info']
    }
    
    with open(os.path.join(output_dir, f"{video_name}_optimal_parameters.json"), 'w') as f:
        json.dump(optimal_params, f, indent=2)
    
    # 2. Save raw frame analysis data as CSV
    raw_data = analysis_results['raw_results']
    df = pd.DataFrame({
        'frame_index': raw_data['frame_indices'],
        'frame_time_seconds': raw_data['frame_times'],
        'crop_width': raw_data['crop_x'],
        'crop_height': raw_data['crop_y'],
        'overlap_width_percent': raw_data['overlap_x'],
        'overlap_height_percent': raw_data['overlap_y']
    })
    df.to_csv(os.path.join(output_dir, f"{video_name}_frame_analysis.csv"), index=False)
    
    # 3. Save complete analysis results as JSON
    # Remove raw_results to avoid duplication (it's in CSV)
    complete_results = analysis_results.copy()
    complete_results.pop('raw_results', None)
    
    with open(os.path.join(output_dir, f"{video_name}_complete_analysis.json"), 'w') as f:
        json.dump(complete_results, f, indent=2, default=str)
    
    # 4. Save summary report as text
    save_summary_report(analysis_results, output_dir, video_name)
    
    # 5. Save plots as image files
    save_analysis_plots(analysis_results, output_dir, video_name)
    
    # 6. Save code snippet for easy copy-paste
    save_code_snippet(analysis_results, output_dir, video_name)
    
    print(f"\n✅ All results saved to: {output_dir}")
    print(f"   📄 {video_name}_optimal_parameters.json - Main parameters")
    print(f"   📊 {video_name}_frame_analysis.csv - Raw frame data") 
    print(f"   📋 {video_name}_complete_analysis.json - Full analysis")
    print(f"   📝 {video_name}_summary_report.txt - Human-readable summary")
    print(f"   📈 {video_name}_analysis_plots.png - Visualization")
    print(f"   💻 {video_name}_code_snippet.py - Ready-to-use code")

def save_summary_report(analysis_results, output_dir, video_name):
    """Save a human-readable summary report."""
    
    report_path = os.path.join(output_dir, f"{video_name}_summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("OPTIMAL CROP PARAMETER ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Video info
        info = analysis_results['analysis_info']
        f.write(f"VIDEO INFORMATION:\n")
        f.write(f"   File: {info['video_name']}\n")
        f.write(f"   Resolution: {info['resolution'][0]}x{info['resolution'][1]}\n")
        f.write(f"   Total Frames: {info['total_frames']}\n")
        f.write(f"   FPS: {info['fps']:.1f}\n")
        f.write(f"   Duration: {info['total_frames']/info['fps']:.1f} seconds\n\n")
        
        # Analysis info
        f.write(f"ANALYSIS SETTINGS:\n")
        f.write(f"   Frames Analyzed: {info['num_samples']}\n")
        f.write(f"   Sampling Method: {info['sampling_method']}\n")
        f.write(f"   Analysis Mode: {info['analysis_mode']}\n")
        f.write(f"   Combination Method: {info['combination_method']}\n")
        f.write(f"   Analysis Date: {info['analysis_timestamp']}\n\n")
        
        # Optimal parameters
        f.write(f"RECOMMENDED PARAMETERS:\n")
        f.write(f"   CROP_WIDTH = {analysis_results['CROP_WIDTH']}\n")
        f.write(f"   CROP_HEIGHT = {analysis_results['CROP_HEIGHT']}\n")
        f.write(f"   OVERLAP_WIDTH_PERCENTAGE = {analysis_results['OVERLAP_WIDTH_PERCENTAGE']}\n")
        f.write(f"   OVERLAP_HEIGHT_PERCENTAGE = {analysis_results['OVERLAP_HEIGHT_PERCENTAGE']}\n\n")
        
        # Method info
        f.write(f"SELECTION METHOD:\n")
        f.write(f"   Method: {analysis_results.get('method', 'unknown')}\n")
        if 'confidence' in analysis_results:
            f.write(f"   Confidence: {analysis_results['confidence']:.2f}\n")
        if 'source_frame' in analysis_results:
            f.write(f"   Source Frame: {analysis_results['source_frame']}\n")
        f.write("\n")
        
        # Statistics if available
        if 'statistics' in analysis_results:
            stats = analysis_results['statistics']
            f.write(f"STATISTICS ACROSS ALL FRAMES:\n")
            f.write(f"   Crop Width  - Mean: {stats['crop_x']['mean']:.0f}, Range: {stats['crop_x']['min']:.0f}-{stats['crop_x']['max']:.0f}\n")
            f.write(f"   Crop Height - Mean: {stats['crop_y']['mean']:.0f}, Range: {stats['crop_y']['min']:.0f}-{stats['crop_y']['max']:.0f}\n")
            f.write(f"   Overlap X%  - Mean: {stats['overlap_x']['mean']:.1f}, Range: {stats['overlap_x']['min']:.0f}-{stats['overlap_x']['max']:.0f}\n")
            f.write(f"   Overlap Y%  - Mean: {stats['overlap_y']['mean']:.1f}, Range: {stats['overlap_y']['min']:.0f}-{stats['overlap_y']['max']:.0f}\n\n")
        
        # Recommendations
        if 'recommendations' in analysis_results:
            f.write(f"RECOMMENDATIONS:\n")
            for rec in analysis_results['recommendations']:
                f.write(f"   {rec}\n")
        
        f.write("\n" + "="*60 + "\n")

def save_analysis_plots(analysis_results, output_dir, video_name):
    """Save analysis plots as image files."""
    
    results = analysis_results['raw_results']
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Crop Parameter Analysis: {video_name}', fontsize=16)
    
    # Crop dimensions over time
    axes[0, 0].plot(results['frame_times'], results['crop_x'], 'b-o', label='Crop Width', alpha=0.7)
    axes[0, 0].plot(results['frame_times'], results['crop_y'], 'r-o', label='Crop Height', alpha=0.7)
    axes[0, 0].axhline(analysis_results['CROP_WIDTH'], color='b', linestyle='--', alpha=0.8, 
                       label=f"Optimal Width: {analysis_results['CROP_WIDTH']}")
    axes[0, 0].axhline(analysis_results['CROP_HEIGHT'], color='r', linestyle='--', alpha=0.8, 
                       label=f"Optimal Height: {analysis_results['CROP_HEIGHT']}")
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Crop Size (pixels)')
    axes[0, 0].set_title('Crop Dimensions Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Overlap percentages over time
    axes[0, 1].plot(results['frame_times'], results['overlap_x'], 'g-o', label='Overlap X%', alpha=0.7)
    axes[0, 1].plot(results['frame_times'], results['overlap_y'], 'm-o', label='Overlap Y%', alpha=0.7)
    axes[0, 1].axhline(analysis_results['OVERLAP_WIDTH_PERCENTAGE'], color='g', linestyle='--', alpha=0.8, 
                       label=f"Optimal X: {analysis_results['OVERLAP_WIDTH_PERCENTAGE']}%")
    axes[0, 1].axhline(analysis_results['OVERLAP_HEIGHT_PERCENTAGE'], color='m', linestyle='--', alpha=0.8, 
                       label=f"Optimal Y: {analysis_results['OVERLAP_HEIGHT_PERCENTAGE']}%")
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Overlap Percentage')
    axes[0, 1].set_title('Overlap Percentages Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histograms of crop sizes
    axes[1, 0].hist(results['crop_x'], bins=10, alpha=0.7, label='Crop Width', color='blue')
    axes[1, 0].hist(results['crop_y'], bins=10, alpha=0.7, label='Crop Height', color='red')
    axes[1, 0].axvline(analysis_results['CROP_WIDTH'], color='blue', linestyle='--', linewidth=2)
    axes[1, 0].axvline(analysis_results['CROP_HEIGHT'], color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Crop Size (pixels)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Crop Sizes')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histograms of overlaps
    axes[1, 1].hist(results['overlap_x'], bins=10, alpha=0.7, label='Overlap X%', color='green')
    axes[1, 1].hist(results['overlap_y'], bins=10, alpha=0.7, label='Overlap Y%', color='magenta')
    axes[1, 1].axvline(analysis_results['OVERLAP_WIDTH_PERCENTAGE'], color='green', linestyle='--', linewidth=2)
    axes[1, 1].axvline(analysis_results['OVERLAP_HEIGHT_PERCENTAGE'], color='magenta', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Overlap Percentage')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Overlap Percentages')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{video_name}_analysis_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_code_snippet(analysis_results, output_dir, video_name):
    """Save a ready-to-use code snippet."""
    
    code_path = os.path.join(output_dir, f"{video_name}_code_snippet.py")
    
    with open(code_path, 'w') as f:
        f.write("# Optimal crop parameters generated from video analysis\n")
        f.write(f"# Video: {video_name}\n")
        f.write(f"# Analysis date: {analysis_results['analysis_info']['analysis_timestamp']}\n")
        f.write(f"# Method: {analysis_results.get('method', 'unknown')}\n\n")
        
        f.write("# Copy these parameters into your main script:\n")
        f.write(f"CROP_WIDTH = {analysis_results['CROP_WIDTH']}\n")
        f.write(f"CROP_HEIGHT = {analysis_results['CROP_HEIGHT']}\n")
        f.write(f"OVERLAP_WIDTH_PERCENTAGE = {analysis_results['OVERLAP_WIDTH_PERCENTAGE']}\n")
        f.write(f"OVERLAP_HEIGHT_PERCENTAGE = {analysis_results['OVERLAP_HEIGHT_PERCENTAGE']}\n\n")
        
        f.write("# Usage in MakeCropsDetectThem:\n")
        f.write("element_crops = MakeCropsDetectThem(\n")
        f.write("    image=image_to_infer_on_bgr,\n")
        f.write("    model=crop_detection_model,\n")
        f.write("    imgsz=MODEL_INPUT_SIZE,\n")
        f.write("    conf=BASE_MODEL_CONF_FOR_CROPS,\n")
        f.write("    iou=POSTPROCESS_IOU_THRESHOLD,\n")
        f.write("    classes_list=None,\n")
        f.write("    segment=False,\n")
        f.write(f"    shape_x=CROP_WIDTH,  # {analysis_results['CROP_WIDTH']}\n")
        f.write(f"    shape_y=CROP_HEIGHT,  # {analysis_results['CROP_HEIGHT']}\n")
        f.write(f"    overlap_x=OVERLAP_WIDTH_PERCENTAGE,  # {analysis_results['OVERLAP_WIDTH_PERCENTAGE']}\n")
        f.write(f"    overlap_y=OVERLAP_HEIGHT_PERCENTAGE,  # {analysis_results['OVERLAP_HEIGHT_PERCENTAGE']}\n")
        f.write("    show_crops=False,\n")
        f.write("    show_processing_status=ENABLE_PROGRESS_TRACKING,\n")
        f.write("    resize_initial_size=True,\n")
        f.write("    memory_optimize=USE_MEMORY_OPTIMIZE,\n")
        f.write("    batch_inference=USE_BATCH_INFERENCE,\n")
        f.write("    inference_extra_args=None\n")
        f.write(")\n")

def calculate_optimal_parameters(results, combination_method, verbose=True):
    """Calculate optimal parameters from collected results."""
    
    crop_x_vals = np.array(results['crop_x'])
    crop_y_vals = np.array(results['crop_y'])
    overlap_x_vals = np.array(results['overlap_x'])
    overlap_y_vals = np.array(results['overlap_y'])
    
    if combination_method == "mean":
        optimal_crop_x = int(np.mean(crop_x_vals))
        optimal_crop_y = int(np.mean(crop_y_vals))
        optimal_overlap_x = int(np.mean(overlap_x_vals))
        optimal_overlap_y = int(np.mean(overlap_y_vals))
        
    elif combination_method == "median":
        optimal_crop_x = int(np.median(crop_x_vals))
        optimal_crop_y = int(np.median(crop_y_vals))
        optimal_overlap_x = int(np.median(overlap_x_vals))
        optimal_overlap_y = int(np.median(overlap_y_vals))
        
    elif combination_method == "mode":
        optimal_crop_x = Counter(crop_x_vals).most_common(1)[0][0]
        optimal_crop_y = Counter(crop_y_vals).most_common(1)[0][0]
        optimal_overlap_x = Counter(overlap_x_vals).most_common(1)[0][0]
        optimal_overlap_y = Counter(overlap_y_vals).most_common(1)[0][0]
        
    elif combination_method == "conservative":
        # Use larger crop sizes (75th percentile) and higher overlaps (75th percentile)
        optimal_crop_x = int(np.percentile(crop_x_vals, 75))
        optimal_crop_y = int(np.percentile(crop_y_vals, 75))
        optimal_overlap_x = int(np.percentile(overlap_x_vals, 75))
        optimal_overlap_y = int(np.percentile(overlap_y_vals, 75))
    
    # Calculate statistics
    stats = {
        'crop_x': {
            'mean': np.mean(crop_x_vals),
            'median': np.median(crop_x_vals),
            'std': np.std(crop_x_vals),
            'min': np.min(crop_x_vals),
            'max': np.max(crop_x_vals),
            'range': np.max(crop_x_vals) - np.min(crop_x_vals)
        },
        'crop_y': {
            'mean': np.mean(crop_y_vals),
            'median': np.median(crop_y_vals),
            'std': np.std(crop_y_vals),
            'min': np.min(crop_y_vals),
            'max': np.max(crop_y_vals),
            'range': np.max(crop_y_vals) - np.min(crop_y_vals)
        },
        'overlap_x': {
            'mean': np.mean(overlap_x_vals),
            'median': np.median(overlap_x_vals),
            'std': np.std(overlap_x_vals),
            'min': np.min(overlap_x_vals),
            'max': np.max(overlap_x_vals),
            'range': np.max(overlap_x_vals) - np.min(overlap_x_vals)
        },
        'overlap_y': {
            'mean': np.mean(overlap_y_vals),
            'median': np.median(overlap_y_vals),
            'std': np.std(overlap_y_vals),
            'min': np.min(overlap_y_vals),
            'max': np.max(overlap_y_vals),
            'range': np.max(overlap_y_vals) - np.min(overlap_y_vals)
        }
    }
    
    # Assess stability
    stability_score = calculate_stability_score(stats)
    
    recommendations = generate_recommendations(stats, stability_score)
    
    optimal_params = {
        'CROP_WIDTH': optimal_crop_x,
        'CROP_HEIGHT': optimal_crop_y,
        'OVERLAP_WIDTH_PERCENTAGE': optimal_overlap_x,
        'OVERLAP_HEIGHT_PERCENTAGE': optimal_overlap_y,
        'combination_method': combination_method,
        'statistics': stats,
        'stability_score': stability_score,
        'recommendations': recommendations,
        'raw_results': results
    }
    
    if verbose:
        print_analysis_summary(optimal_params)
    
    return optimal_params

def calculate_stability_score(stats):
    """Calculate how stable/consistent the parameters are across frames."""
    # Lower coefficient of variation = more stable
    cv_crop_x = stats['crop_x']['std'] / stats['crop_x']['mean']
    cv_crop_y = stats['crop_y']['std'] / stats['crop_y']['mean']
    cv_overlap_x = stats['overlap_x']['std'] / max(stats['overlap_x']['mean'], 1)
    cv_overlap_y = stats['overlap_y']['std'] / max(stats['overlap_y']['mean'], 1)
    
    avg_cv = (cv_crop_x + cv_crop_y + cv_overlap_x + cv_overlap_y) / 4
    
    # Convert to stability score (0-100, higher = more stable)
    stability_score = max(0, 100 - (avg_cv * 100))
    
    return stability_score

def generate_recommendations(stats, stability_score):
    """Generate recommendations based on the analysis."""
    recommendations = []
    
    if stability_score > 80:
        recommendations.append("✅ High stability - parameters are consistent across frames")
    elif stability_score > 60:
        recommendations.append("⚠️ Moderate stability - some variation in optimal parameters")
    else:
        recommendations.append("❌ Low stability - significant variation, consider manual tuning")
    
    # Check crop size variability
    if stats['crop_x']['range'] > stats['crop_x']['mean'] * 0.5:
        recommendations.append("⚠️ High crop width variation - video content varies significantly")
    
    if stats['crop_y']['range'] > stats['crop_y']['mean'] * 0.5:
        recommendations.append("⚠️ High crop height variation - video content varies significantly")
    
    # Check overlap patterns
    if stats['overlap_x']['mean'] < 10:
        recommendations.append("📝 Low overlap detected - objects may be cut at boundaries")
    
    if stats['overlap_x']['mean'] > 50:
        recommendations.append("📝 High overlap detected - processing will be slower but more thorough")
    
    return recommendations

def plot_analysis_results(results, analysis_results):
    """Create visualizations of the analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Video Crop Parameter Analysis', fontsize=16)
    
    # Crop dimensions over time
    axes[0, 0].plot(results['frame_times'], results['crop_x'], 'b-o', label='Crop Width', alpha=0.7)
    axes[0, 0].plot(results['frame_times'], results['crop_y'], 'r-o', label='Crop Height', alpha=0.7)
    axes[0, 0].axhline(analysis_results['CROP_WIDTH'], color='b', linestyle='--', alpha=0.8, label=f"Optimal Width: {analysis_results['CROP_WIDTH']}")
    axes[0, 0].axhline(analysis_results['CROP_HEIGHT'], color='r', linestyle='--', alpha=0.8, label=f"Optimal Height: {analysis_results['CROP_HEIGHT']}")
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Crop Size (pixels)')
    axes[0, 0].set_title('Crop Dimensions Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Overlap percentages over time
    axes[0, 1].plot(results['frame_times'], results['overlap_x'], 'g-o', label='Overlap X%', alpha=0.7)
    axes[0, 1].plot(results['frame_times'], results['overlap_y'], 'm-o', label='Overlap Y%', alpha=0.7)
    axes[0, 1].axhline(analysis_results['OVERLAP_WIDTH_PERCENTAGE'], color='g', linestyle='--', alpha=0.8, label=f"Optimal X: {analysis_results['OVERLAP_WIDTH_PERCENTAGE']}%")
    axes[0, 1].axhline(analysis_results['OVERLAP_HEIGHT_PERCENTAGE'], color='m', linestyle='--', alpha=0.8, label=f"Optimal Y: {analysis_results['OVERLAP_HEIGHT_PERCENTAGE']}%")
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Overlap Percentage')
    axes[0, 1].set_title('Overlap Percentages Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histograms of crop sizes
    axes[1, 0].hist(results['crop_x'], bins=10, alpha=0.7, label='Crop Width', color='blue')
    axes[1, 0].hist(results['crop_y'], bins=10, alpha=0.7, label='Crop Height', color='red')
    axes[1, 0].axvline(analysis_results['CROP_WIDTH'], color='blue', linestyle='--', linewidth=2)
    axes[1, 0].axvline(analysis_results['CROP_HEIGHT'], color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Crop Size (pixels)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Crop Sizes')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histograms of overlaps
    axes[1, 1].hist(results['overlap_x'], bins=10, alpha=0.7, label='Overlap X%', color='green')
    axes[1, 1].hist(results['overlap_y'], bins=10, alpha=0.7, label='Overlap Y%', color='magenta')
    axes[1, 1].axvline(analysis_results['OVERLAP_WIDTH_PERCENTAGE'], color='green', linestyle='--', linewidth=2)
    axes[1, 1].axvline(analysis_results['OVERLAP_HEIGHT_PERCENTAGE'], color='magenta', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Overlap Percentage')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Overlap Percentages')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_analysis_summary(optimal_params):
    """Print a formatted summary of the analysis results."""
    print("\n" + "="*60)
    print("OPTIMAL CROP PARAMETER ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\n📋 RECOMMENDED PARAMETERS:")
    print(f"CROP_WIDTH = {optimal_params['CROP_WIDTH']}")
    print(f"CROP_HEIGHT = {optimal_params['CROP_HEIGHT']}")
    print(f"OVERLAP_WIDTH_PERCENTAGE = {optimal_params['OVERLAP_WIDTH_PERCENTAGE']}")
    print(f"OVERLAP_HEIGHT_PERCENTAGE = {optimal_params['OVERLAP_HEIGHT_PERCENTAGE']}")
    
    print(f"\n📊 STABILITY SCORE: {optimal_params['stability_score']:.1f}/100")
    
    print(f"\n📈 STATISTICS SUMMARY:")
    stats = optimal_params['statistics']
    print(f"Crop Width  - Mean: {stats['crop_x']['mean']:.0f}, Range: {stats['crop_x']['min']:.0f}-{stats['crop_x']['max']:.0f}, Std: {stats['crop_x']['std']:.1f}")
    print(f"Crop Height - Mean: {stats['crop_y']['mean']:.0f}, Range: {stats['crop_y']['min']:.0f}-{stats['crop_y']['max']:.0f}, Std: {stats['crop_y']['std']:.1f}")
    print(f"Overlap X%  - Mean: {stats['overlap_x']['mean']:.1f}, Range: {stats['overlap_x']['min']:.0f}-{stats['overlap_x']['max']:.0f}, Std: {stats['overlap_x']['std']:.1f}")
    print(f"Overlap Y%  - Mean: {stats['overlap_y']['mean']:.1f}, Range: {stats['overlap_y']['min']:.0f}-{stats['overlap_y']['max']:.0f}, Std: {stats['overlap_y']['std']:.1f}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in optimal_params['recommendations']:
        print(f"   {rec}")
    
    print("\n" + "="*60)
    
def print_method_comparison(method_comparison):
    """Print a formatted comparison of all methods."""
    
    print(f"\n" + "="*80)
    print("METHOD COMPARISON SUMMARY")
    print("="*80)
    
    # Create formatted table
    print(f"{'Method':<20} {'Crop W':<8} {'Crop H':<8} {'Ovlp W%':<8} {'Ovlp H%':<8} {'Conf':<8} {'Stab':<8}")
    print("-"*80)
    
    for result in method_comparison:
        print(f"{result['method']:<20} {result['crop_width']:<8} {result['crop_height']:<8} "
              f"{result['overlap_width']:<8} {result['overlap_height']:<8} "
              f"{result['confidence']:<8.3f} {result['stability_score']:<8.1f}")
    
    print("="*80)

if __name__ == "__main__":
    # Detailed analysis with custom parameters
    results = analyze_video_for_optimal_crops_multi_method(
        video_path=r"C:\Users\swest\Desktop\traffic-capturing\sample_videos\5 min test - Large FOV.mp4",
        num_samples=2500,  # More samples = better accuracy
        sampling_method="uniform",  # or "random", "adaptive"
        analysis_mode="network_based",  # or "resolution_based"
        model_path=r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\models\weights\silent_contract_9082_original_640px_bs416.pt",
        combination_methods=["exact_mode","similarity_clustering","centroid_distance","stability_weighted","conservative_config"],
        save_results=True,
        verbose=False,
        conf_threshold=0.1
    )
    
    # Print the code you can copy-paste into your main script
    print(f"\nAnalysis complete! Check the output directory for individual method results and comparison.")

