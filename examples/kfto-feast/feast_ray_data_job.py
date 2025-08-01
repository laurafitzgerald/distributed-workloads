#!/usr/bin/env python3
"""
Feast + Ray Data Processing Job

This script demonstrates how to use Ray Data for distributed processing
of Feast feature data, converting it to JSONL format for LLM training.

Usage:
    python feast_ray_data_job.py

This script can be submitted as a Ray job to run on a Ray cluster.
"""

import ray
import ray.data as rd
import pandas as pd
import json
import os
from datetime import datetime
from feast import FeatureStore

def main():
    """Main processing function that runs the Feast + Ray Data pipeline."""
    
    print("Starting Feast + Ray Data processing job...")
    
    
    try:
        # Step 1: Initialize Feast feature store
        print("Initializing Feast feature store...")
        store = FeatureStore(repo_path=".")
        
        # Step 2: Define entity data for feature retrieval
        print("Defining entity data for feature retrieval...")
        entity_df = pd.DataFrame.from_dict({
            "driver_id": [1001, 1002, 1003, 1004, 1005],
            "event_timestamp": [
                datetime(2021, 4, 12, 10, 59, 42),
                datetime(2021, 4, 12, 8, 12, 10),
                datetime(2021, 4, 12, 16, 40, 26),
                datetime(2021, 4, 12, 12, 30, 0),
                datetime(2021, 4, 12, 14, 15, 30)
            ],
            "label_driver_reported_satisfaction": [1, 5, 3, 4, 2],
            "val_to_add": [1, 2, 3, 4, 5],
            "val_to_add_2": [10, 20, 30, 40, 50],
        })
        
        # Step 3: Retrieve historical features from Feast
        print("Retrieving historical features from Feast...")
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "driver_hourly_stats:conv_rate",
                "driver_hourly_stats:acc_rate",
                "driver_hourly_stats:avg_daily_trips",
                "transformed_conv_rate:conv_rate_plus_val1",
                "transformed_conv_rate:conv_rate_plus_val2",
            ],
        ).to_df()
        
        print(f"Retrieved {len(training_df)} feature records from Feast")
        print("Feature schema:")
        print(training_df.info())
        
        # Step 4: Convert to Ray Dataset for distributed processing
        print("Converting Feast DataFrame to Ray Dataset...")
        ds = rd.from_pandas(training_df)
        print(f"Created Ray Dataset with {ds.count()} rows")
        
        # Step 5: Transform to document format using Ray Data
        print("Transforming to document format using Ray Data...")
        def create_document(row):
            """Convert a row to a document format for LLM training."""
            return {
                "driver_id": int(row['driver_id']),
                "conv_rate": float(row['conv_rate']),
                "acc_rate": float(row['acc_rate']),
                "avg_daily_trips": int(row['avg_daily_trips']),
            }
        
        documents_ds = ds.map(create_document)
        print("Transformed to document format using Ray Data")
        
        # Step 6: Save documents as JSONL using Ray Data
        stats_document_jsonl_file = "driver_stats_documents.jsonl"
        print(f"Saving documents to {stats_document_jsonl_file}...")
        documents_ds.write_json(stats_document_jsonl_file)
        print(f"Documents saved to {stats_document_jsonl_file}")
        
        # Step 7: Generate training examples using Ray Data
        print("Generating training examples using Ray Data...")
        def create_training_example(record):
            """Create a training example for LLM fine-tuning."""
            instruction_text = "Summarize the driver's performance metrics."
            
            input_text = (
                f"Driver ID: {record['driver_id']}, "
                f"Conversion Rate: {record['conv_rate']:.4f}, "
                f"Acceleration Rate: {record['acc_rate']:.4f}, "
                f"Average Daily Trips: {record['avg_daily_trips']}"
            )
            
            output_text = (
                f"Driver {record['driver_id']} has a conversion rate of {record['conv_rate']:.2%}, "
                f"an acceleration rate of {record['acc_rate']:.2%}, and completes an average of {record['avg_daily_trips']} daily trips."
            )
            
            return {
                "instruction": instruction_text,
                "input": input_text,
                "output": output_text
            }
        
        training_ds = documents_ds.map(create_training_example)
        
        # Step 8: Save training examples as JSONL
        output_file = "driver_stats_training.jsonl"
        print(f"Saving training examples to {output_file}...")
        training_ds.write_json(output_file)
        print(f"Training examples saved to {output_file}")
        
        # Step 9: Verify the output
        print("Verifying output...")
        verification_ds = rd.read_json(output_file)
        final_count = verification_ds.count()
        print(f"Generated {final_count} training examples")
        
        # Show a sample of the generated training data
        print("\nSample training example:")
        sample = verification_ds.take(1)[0]
        print(json.dumps(sample, indent=2))
        
        # Step 10: Optional - Enhanced processing with Feast integration
        print("\nPerforming enhanced processing with Feast integration...")
        def enrich_with_feast_features(batch):
            """Enrich a batch of entity data with Feast features."""
            batch_df = pd.DataFrame(batch)
            
            # Get features from Feast
            features = store.get_historical_features(
                entity_df=batch_df,
                features=[
                    "driver_hourly_stats:conv_rate",
                    "driver_hourly_stats:acc_rate",
                    "driver_hourly_stats:avg_daily_trips",
                    "transformed_conv_rate:conv_rate_plus_val1",
                    "transformed_conv_rate:conv_rate_plus_val2",
                ],
            ).to_df()
            
            return features.to_dict('records')
        
        # Process data in batches using Ray Data
        enriched_ds = ds.map_batches(enrich_with_feast_features, batch_size=2)
        
        # Create enhanced training examples
        def flatten_and_create_examples(batch):
            """Flatten batch results and create training examples."""
            examples = []
            for record in batch:
                example = create_training_example(record)
                examples.append(example)
            return examples
        
        final_training_ds = enriched_ds.map_batches(flatten_and_create_examples)
        
        # Save enhanced training data
        enhanced_output_file = "enhanced_driver_stats_training.jsonl"
        final_training_ds.write_json(enhanced_output_file)
        
        enhanced_count = final_training_ds.count()
        print(f"Generated {enhanced_count} enhanced training examples")
        
        # Step 11: Summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Files generated:")
        print(f"  - {stats_document_jsonl_file}: {documents_ds.count()} documents")
        print(f"  - {output_file}: {final_count} training examples")
        print(f"  - {enhanced_output_file}: {enhanced_count} enhanced training examples")
        print("="*50)
        
        # List generated files
        generated_files = [stats_document_jsonl_file, output_file, enhanced_output_file]
        for file in generated_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"✓ {file} ({size} bytes)")
            else:
                print(f"✗ {file} (not found)")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    
    finally:
        # Cleanup Ray resources
        print("Done")

if __name__ == "__main__":
    main() 