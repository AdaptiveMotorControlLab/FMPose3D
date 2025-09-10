from JSONProcessor import JSONProcessor
import os
import datetime

if __name__ == "__main__":
    # Create a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"image_counts_{timestamp}.txt"
    
    # Create an empty file to start with
    with open(output_file, 'w') as f:
        f.write(f"Image counts generated on {timestamp}\n\n")
    
    dataset_list = ["train", "test", "val"]
    # dataset_list = ["test"]
    
    for dataset in dataset_list:
        # Always use append mode ('a') to avoid overwriting
        with open(output_file, 'a') as f:
            f.write(f"\n{'='*20} {dataset.upper()} DATASET {'='*20}\n")
            f.write("-" * 50 + "\n")
            
            base_dir = f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/splitted_{dataset}_datasets"
            
            if not os.path.exists(base_dir):
                f.write(f"Directory not found: {base_dir}\n")
                print(f"Directory not found: {base_dir}")
                continue
                
            file_list = os.listdir(base_dir)
            
            total_images = 0
            for file in file_list:
                json_path = os.path.join(base_dir, file)
                
                # Use the static method directly
                num_images = JSONProcessor.cal_number_of_images(json_path)
                
                # Write to file and also print to console
                output_line = f"{file}: {num_images}"
                f.write(output_line + "\n")
                print(output_line)
                
                total_images += num_images
            
            # Add a summary line
            summary = f"\nTotal images for {dataset} dataset: {total_images}\n"
            f.write(summary)
            f.write("-" * 50 + "\n")
            print(summary)
        
        print(f"Results for {dataset} dataset added to {output_file}")
    
    print(f"\nAll results saved to {output_file}")
