import pandas as pd
import numpy as np
from PIL import Image
import json
import os
import glob
from shapely.geometry import Point, Polygon, box
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

def process_single_file(csv_file, aoi_folder, image_folder):
    """
    Process a single eye-tracking CSV file and match it with corresponding AOI data
    
    Args:
        csv_file: Path to the CSV file with eye-tracking data
        aoi_folder: Path to the folder containing AOI JSON files
        image_folder: Path to the folder containing image files
        
    Returns:
        dict: Dictionary containing processed AOI statistics
    """
    print(f"Processing {os.path.basename(csv_file)}...")
    
    # Read CSV data
    data = pd.read_csv(csv_file)
    
    # Get the LOD value (assuming it's consistent in the file)
    lod_values = data['LOD'].unique()
    if len(lod_values) == 0:
        print(f"No LOD values found in {csv_file}, skipping...")
        return None
    
    results = {}
    
    # Process each unique LOD value in the file
    for lod in lod_values:
        # Skip if LOD is NaN or empty
        if pd.isna(lod) or lod == "":
            continue
            
        print(f"  Processing LOD: {lod}")
        
        # Filter data for current LOD
        filtered_data = data.loc[data['LOD'] == lod]
        
        # Get hit UV coordinates
        hitUV_list = filtered_data['hitUV'].tolist()
        
        # Find corresponding image file
        image_path = None
        
        # Look for an image file matching the LOD value
        image_files = glob.glob(os.path.join(image_folder, f"*{lod}*.jpg"))
        if not image_files:
            print(f"  Warning: No matching image found for LOD '{lod}'")
            continue
            
        image_path = image_files[0]
        image_filename = os.path.basename(image_path)
        
        # Find corresponding AOI file
        aoi_files = glob.glob(os.path.join(aoi_folder, f"*{lod.replace('-', '')}*.json"))
        
        if not aoi_files:
            print(f"  Warning: No matching AOI file found for LOD '{lod}'")
            continue
            
        aoi_path = aoi_files[0]
        
        # Process coordinates
        coordinates = []
        for coord in hitUV_list:
            try:
                x, y = map(float, coord.strip('()').split(','))
                # Apply correction factors
                x = x + 0.025
                y = y - 0.16
                # Ensure coordinates are within valid range
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                coordinates.append((x, y))
            except ValueError as e:
                # Skip invalid coordinates
                continue
        
        # If no valid coordinates, skip
        if not coordinates:
            print(f"  Warning: No valid coordinates found for LOD '{lod}'")
            continue
        
        # Read image and get dimensions
        try:
            image = Image.open(image_path)
            width, height = image.size
        except Exception as e:
            print(f"  Error opening image {image_path}: {e}")
            continue
        
        # Convert UV coordinates to pixel coordinates
        pixels = [(int(x * width), int(y * height)) for x, y in coordinates]
        
        # Read AOI data
        try:
            with open(aoi_path, 'r') as f:
                aoi_data = json.load(f)
        except Exception as e:
            print(f"  Error reading AOI file {aoi_path}: {e}")
            continue
        
        # Get AOI information
        # Check different possible keys in the JSON structure
        img_metadata = aoi_data.get('_via_img_metadata', {})
        
        # Find the correct key for the image
        aoi_info = None
        for key in img_metadata:
            if image_filename in key or any(key in img_path for img_path in image_files):
                aoi_info = img_metadata[key].get('regions', [])
                break
        
        # If we couldn't find a matching entry, try a fallback approach
        if aoi_info is None:
            # Just use the first entry if available
            if img_metadata and len(img_metadata) > 0:
                first_key = list(img_metadata.keys())[0]
                aoi_info = img_metadata[first_key].get('regions', [])
            else:
                print(f"  Warning: Could not find AOI information for {image_filename}")
                continue
        
        # Initialize data structures
        aoi_distribution = defaultdict(list)
        background_points = set(pixels)
        
        # Create a figure for visualization
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        # Process each AOI
        for aoi in aoi_info:
            shape_attributes = aoi.get('shape_attributes', {})
            region_attributes = aoi.get('region_attributes', {})
            
            # Skip if necessary data is missing
            if not shape_attributes or 'name' not in shape_attributes:
                continue
            
            aoi_type = region_attributes.get('type', None)
            if aoi_type is None:
                aoi_type = 'undefined'  # 当type字段完全不存在时
            elif not str(aoi_type).strip():
                aoi_type = 'undefined'  # 当type字段为空字符串时
            else:
                aoi_type = str(aoi_type).strip().lower()  # 保留其他有效值，包括'unknown'

            contained_points = []
            
            # Process according to shape type
            if shape_attributes['name'] == 'polygon':
                # Get polygon points
                all_points_x = shape_attributes.get('all_points_x', [])
                all_points_y = shape_attributes.get('all_points_y', [])
                
                if not all_points_x or not all_points_y or len(all_points_x) != len(all_points_y):
                    continue
                
                # Create polygon
                try:
                    polygon = Polygon(zip(all_points_x, all_points_y))
                    
                    # Draw polygon
                    poly_patch = patches.Polygon(
                        list(zip(all_points_x, all_points_y)), 
                        closed=True, 
                        edgecolor='r', 
                        facecolor='none',
                        linewidth=1.5,
                        alpha=0.7
                    )
                    ax.add_patch(poly_patch)
                    
                    # Check which points are inside the polygon
                    contained_points = [pixel for pixel in pixels if polygon.contains(Point(pixel))]
                except Exception as e:
                    print(f"  Error processing polygon: {e}")
                    continue
                    
            elif shape_attributes['name'] == 'rect':
                # Get rectangle coordinates
                try:
                    x = shape_attributes.get('x', 0)
                    y = shape_attributes.get('y', 0)
                    rect_width = shape_attributes.get('width', 0)
                    rect_height = shape_attributes.get('height', 0)
                    
                    # Create rectangle
                    rectangle = box(x, y, x + rect_width, y + rect_height)
                    
                    # Draw rectangle
                    rect_patch = patches.Rectangle(
                        (x, y), 
                        rect_width, 
                        rect_height, 
                        edgecolor='b', 
                        facecolor='none',
                        linewidth=1.5,
                        alpha=0.7
                    )
                    ax.add_patch(rect_patch)
                    
                    # Check which points are inside the rectangle
                    contained_points = [pixel for pixel in pixels if rectangle.contains(Point(pixel))]
                except Exception as e:
                    print(f"  Error processing rectangle: {e}")
                    continue
            
            # If we found points in this AOI
            if contained_points:
                # Add to the distribution counts
                aoi_distribution[aoi_type].extend(contained_points)
                # Add labels to the AOIs
                centroid_x = np.mean([p[0] for p in contained_points]) if contained_points else 0
                centroid_y = np.mean([p[1] for p in contained_points]) if contained_points else 0
                ax.text(centroid_x, centroid_y, aoi_type, color='blue', fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.7))
                # Remove from background points
                background_points -= set(contained_points)
        
        # Add background points
        if background_points:
            aoi_distribution['Background'] = list(background_points)
            
        # Plot eye tracking points
        for i, pixel in enumerate(pixels):
            if i < len(hitUV_list):  # Only add points that have corresponding data
                # Color code points based on their AOI type
                for aoi_type, points in aoi_distribution.items():
                    if pixel in points:
                        color = {'Background': 'gray'}.get(aoi_type, 'green')
                        alpha = 0.7
                        break
                else:
                    color = 'yellow'
                    alpha = 0.5
                    
                ax.plot(pixel[0], pixel[1], 'o', color=color, markersize=5, alpha=alpha)
        
        # Add title and save visualization
        plt.title(f"Eye Tracking for LOD: {lod}")
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(csv_file), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save the image
        output_filename = f"{os.path.splitext(os.path.basename(csv_file))[0]}_{lod}_aoi_map.png"
        output_path = os.path.join(results_dir, output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"  Saved visualization to: {output_path}")
        
        # Compute statistics for this LOD
        aoi_stats = {}
        total_points = len(pixels)
        
        for aoi_type, points in aoi_distribution.items():
            point_count = len(points)
            percentage = (point_count / total_points * 100) if total_points > 0 else 0
            aoi_stats[aoi_type] = {
                'count': point_count,
                'percentage': percentage
            }
        
        # Store results for this LOD
        results[lod] = {
            'aoi_distribution': dict(aoi_distribution),
            'aoi_stats': aoi_stats,
            'total_points': total_points
        }
    
    return results

def visualize_aoi_distribution(all_results, results_dir):
    """
    Create visualizations of AOI distribution across all processed files
    
    Args:
        all_results: Dictionary containing results from all processed files
        results_dir: Directory for saving results
    """
    print("Creating AOI distribution visualizations...")
    os.makedirs(results_dir, exist_ok=True)
    
    # Aggregate data by LOD
    lod_aggregated = defaultdict(lambda: defaultdict(list))
    
    for file_name, lod_results in all_results.items():
        for lod, results in lod_results.items():
            for aoi_type, stats in results['aoi_stats'].items():
                lod_aggregated[lod][aoi_type].append(stats['percentage'])
    
    # Create visualization for each LOD
    for lod, aoi_data in lod_aggregated.items():
        # Compute mean and standard deviation for each AOI type
        aoi_means = {}
        aoi_stds = {}
        
        for aoi_type, percentages in aoi_data.items():
            aoi_means[aoi_type] = np.mean(percentages)
            aoi_stds[aoi_type] = np.std(percentages)
        
        # Sort AOI types by mean percentage (descending)
        sorted_aoi_types = sorted(aoi_means.keys(), 
                         key=lambda x: (aoi_means[x], x != 'undefined'), 
                         reverse=True)
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        
        x_pos = np.arange(len(sorted_aoi_types))
        means = [aoi_means[aoi] for aoi in sorted_aoi_types]
        stds = [aoi_stds[aoi] for aoi in sorted_aoi_types]
        
        bars = plt.bar(x_pos, means, yerr=stds, align='center', 
                       alpha=0.7, ecolor='black', capsize=10)
        
        # Add values on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{means[i]:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('AOI Type')
        plt.ylabel('Percentage of Fixations (%)')
        plt.title(f'Distribution of Eye Fixations Across AOI Types - LOD: {lod}')
        plt.xticks(x_pos, sorted_aoi_types, rotation=45, ha='right')
        plt.ylim(0, max(means) * 1.2)  # Add some space for the error bars and text
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(results_dir, f"aoi_distribution_{lod}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"  Saved distribution chart to: {output_path}")
    
    # Create aggregated visualization across all LODs
    all_aoi_data = defaultdict(list)
    
    for lod, aoi_data in lod_aggregated.items():
        for aoi_type, percentages in aoi_data.items():
            all_aoi_data[aoi_type].extend(percentages)
    
    # Compute overall mean and standard deviation
    overall_means = {}
    overall_stds = {}
    
    for aoi_type, percentages in all_aoi_data.items():
        overall_means[aoi_type] = np.mean(percentages)
        overall_stds[aoi_type] = np.std(percentages)
    
    # Sort AOI types by mean percentage (descending)
    sorted_aoi_types = sorted(overall_means.keys(), 
                         key=lambda x: (overall_means[x], x != 'undefined'), 
                         reverse=True)
    
    # Create bar chart for overall distribution
    plt.figure(figsize=(14, 7))
    
    x_pos = np.arange(len(sorted_aoi_types))
    means = [overall_means[aoi] for aoi in sorted_aoi_types]
    stds = [overall_stds[aoi] for aoi in sorted_aoi_types]
    
    bars = plt.bar(x_pos, means, yerr=stds, align='center', 
                   alpha=0.7, ecolor='black', capsize=10)
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{means[i]:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('AOI Type', fontsize=12)
    plt.ylabel('Percentage of Fixations (%)', fontsize=12)
    plt.title('Overall Distribution of Eye Fixations Across AOI Types (All LODs)', fontsize=14)
    plt.xticks(x_pos, sorted_aoi_types, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, max(means) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the overall plot
    output_path = os.path.join(results_dir, "overall_aoi_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"  Saved overall distribution chart to: {output_path}")
    
    # Create a heatmap for visualization of AOI distribution across LODs
    if len(lod_aggregated) > 1:
        # Get all unique AOI types
        all_aoi_types = set()
        for lod_data in lod_aggregated.values():
            all_aoi_types.update(lod_data.keys())
        
        # Sort AOI types by overall mean
        sorted_aoi_types = sorted(all_aoi_types, 
                                 key=lambda x: overall_means.get(x, 0), 
                                 reverse=True)
        
        # Sort LODs alphabetically
        sorted_lods = sorted(lod_aggregated.keys())
        
        # Create data matrix for heatmap
        heatmap_data = np.zeros((len(sorted_aoi_types), len(sorted_lods)))
        
        for i, aoi_type in enumerate(sorted_aoi_types):
            for j, lod in enumerate(sorted_lods):
                if aoi_type in lod_aggregated[lod]:
                    heatmap_data[i, j] = np.mean(lod_aggregated[lod][aoi_type])
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        plt.imshow(heatmap_data, cmap='YlOrRd')
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Mean Percentage of Fixations (%)')
        
        # Add labels
        plt.xticks(np.arange(len(sorted_lods)), sorted_lods, rotation=45, ha='right')
        plt.yticks(np.arange(len(sorted_aoi_types)), sorted_aoi_types)
        
        # Add values in cells
        for i in range(len(sorted_aoi_types)):
            for j in range(len(sorted_lods)):
                value = heatmap_data[i, j]
                if value > 0:
                    text_color = 'black' if value < 30 else 'white'
                    plt.text(j, i, f'{value:.1f}%', 
                             ha='center', va='center', 
                             color=text_color, fontsize=9)
        
        plt.xlabel('LOD')
        plt.ylabel('AOI Type')
        plt.title('Distribution of Eye Fixations Across AOI Types and LODs')
        plt.tight_layout()
        
        # Save the heatmap
        output_path = os.path.join(results_dir, "aoi_distribution_heatmap.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"  Saved heatmap visualization to: {output_path}")

def export_results_to_csv(all_results, results_dir):
    """
    Export the results to CSV files for further analysis
    
    Args:
        all_results: Dictionary containing results from all processed files
        results_dir: Directory for saving results
    """
    print("Exporting results to CSV...")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create summary dataframe
    summary_rows = []
    
    for file_name, lod_results in all_results.items():
        for lod, results in lod_results.items():
            for aoi_type, stats in results['aoi_stats'].items():
                summary_rows.append({
                    'File': file_name,
                    'LOD': lod,
                    'AOI_Type': aoi_type,
                    'Point_Count': stats['count'],
                    'Percentage': stats['percentage'],
                    'Total_Points': results['total_points']
                })
    
    # Create dataframe and save to CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        output_path = os.path.join(results_dir, "aoi_summary.csv")
        summary_df.to_csv(output_path, index=False)
        print(f"  Saved summary CSV to: {output_path}")
        
        # Also create a pivot table summary
        pivot_df = summary_df.pivot_table(
            index=['File', 'LOD'],
            columns='AOI_Type',
            values='Percentage',
            aggfunc='first',
            fill_value=0
        )
        
        pivot_output_path = os.path.join(results_dir, "aoi_pivot_summary.csv")
        pivot_df.to_csv(pivot_output_path)
        print(f"  Saved pivot table summary to: {pivot_output_path}")
    else:
        print("  No data to export to CSV")

def main():
    # Define paths
    aoi_folder = "D:\\Yunlei_Data\\LOD_Scene\\AOI_fake"
    image_folder = "D:\\Yunlei_Data\\LOD_Scene\\images"
    csv_folder = "D:\\Yunlei_Data\\LOD_Scene\\real_experiment_data"
    results_dir = os.path.join(csv_folder, "results_fake")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the specified directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files.")
    
    # Process each file
    all_results = {}
    
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        results = process_single_file(csv_file, aoi_folder, image_folder)
        
        if results:
            all_results[file_name] = results
    
    if not all_results:
        print("No results generated from any of the CSV files.")
        return
    
    # Create visualizations
    visualize_aoi_distribution(all_results, results_dir)
    
    # Export results to CSV
    export_results_to_csv(all_results, results_dir)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()