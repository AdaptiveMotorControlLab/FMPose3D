import h5py
import pandas as pd
import numpy as np

def analyze_dlc_h5(file_path):
    """
    Analyze DeepLabCut H5 file structure and content
    """
    print(f"Analyzing DLC H5 file: {file_path}\n")
    
    try:
        # Read the DLC H5 file
        df = pd.read_hdf(file_path)
        
        # Show the multi-index column structure
        print("Multi-Index Column Structure:")
        print("----------------------------")
        print("\nColumn levels:")
        for i in range(len(df.columns.levels)):
            print(f"Level {i} name: {df.columns.names[i]}")
            unique_values = df.columns.get_level_values(i).unique()
            print(f"Unique values: {list(unique_values)}")
            print(f"Count: {len(unique_values)}\n")

        # Show an example of full column names
        print("\nExample of full column names (first 3):")
        print("---------------------------------------")
        for col in list(df.columns)[:3]:
            print(col)
        
        # Basic information
        print("File Structure Overview:")
        print("-----------------------")
        print(f"Number of frames: {len(df)}")
        print(f"Shape: {df.shape}")
        
        # Analyze column structure
        print("\nColumn Structure:")
        print("----------------")
        # Get unique scorers (usually 'dlc')
        scorers = df.columns.get_level_values(0).unique()
        # Get unique individuals (usually 'mus1' in your case)
        individuals = df.columns.get_level_values(1).unique()
        # Get unique bodyparts
        bodyparts = df.columns.get_level_values(2).unique()
        # Get coordinates (x,y,likelihood)
        coords = df.columns.get_level_values(3).unique()
        
        # data = df.columns.get_level_values(4).unique()
        
        print(f"Scorers: {list(scorers)}")
        print(f"Individuals: {list(individuals)}")
        print(f"Bodyparts: {list(bodyparts)}")
        print(f"Coordinates: {list(coords)}")
        # print
        
        # Show data structure
        print("\nData Structure Example:")
        print("----------------------")
        print(df.head())
        
        # Calculate some basic statistics
        print("\nBasic Statistics:")
        print("----------------")
        print(f"Total frames: {len(df)}")
        print(f"Number of tracked points per frame: {len(bodyparts)}")
        
        return df
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

if __name__ == "__main__":
    file_path = "/home/ti_wang/Ti_workspace/PrimatePose/clustering/CollectedData_dlc.h5"
    df = analyze_dlc_h5(file_path)
    