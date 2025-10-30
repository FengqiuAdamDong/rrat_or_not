import asyncio
import json
from motor.motor_asyncio import AsyncIOMotorClient

program = "september_2025_catalog2_revertedRFIsifter"

client = AsyncIOMotorClient(
    host="localhost",
    port=27017,
    username="",
    password="",
)

async def get_events_by_program(client, program_name: str):
    try:
        uuids = await client.frb_master.injections.find({"injection_program": program_name}, {"_id": 0, "id": 1}).to_list(None)
        uuids = [uuid["id"] for uuid in uuids]
        injections = await client.frb_master.injections.find({"id": {"$in": uuids}}, {"_id": 0}).to_list(None)

        # For each unique det_id, return the detection that has the highest combined_snr
        # If one det_id corresponds to multiple detection data entries, the oldest one is returned
        detections = await client.frb_master.detections.aggregate(
            [
                {"$match": {"det_id": {"$in": uuids}}},
                {"$sort": {"combined_snr": -1}},
                # keep all the fields in detection data by using "$$ROOT"
                {
                    "$group": {
                        "_id": "$det_id",
                        "document": {"$first": "$$ROOT"},
                    }
                },
                {"$replaceRoot": {"newRoot": "$document"}},
                {
                    "$project": {
                        "_id": 0,
                    }
                },
            ],
            allowDiskUse=True
        ).to_list(None)
        response = {"injections": injections, "detections": detections}
        return response
    except Exception as e:
        print(e)

async def get_injections_by_program(client, program_name: str):
    try:
        uuids = await client.frb_master.injections.find({"injection_program": program_name}, {"_id": 0, "id": 1}).to_list(None)
        uuids = [uuid["id"] for uuid in uuids]
        injections = await client.frb_master.injections.find({"id": {"$in": uuids}}, {"_id": 0}).to_list(None)

        # For each unique det_id, return the detection that has the highest combined_snr
        # If one det_id corresponds to multiple detection data entries, the oldest one is returned
        detections = await client.frb_master.detections.aggregate(
            [
                {"$match": {"det_id": {"$in": uuids}}},
                {"$sort": {"combined_snr": -1}},
                # keep all the fields in detection data by using "$$ROOT"
                {
                    "$group": {
                        "_id": "$det_id",
                        "document": {"$first": "$$ROOT"},
                    }
                },
                {"$replaceRoot": {"newRoot": "$document"}},
                {
                    "$project": {
                        "_id": 0,
                    }
                },
            ],
            allowDiskUse=True
        ).to_list(None)
        response = {"injections": injections}
        return response
    except Exception as e:
        print(e)

async def batch_fetch_and_save_injections(client, program_name: str, file_path: str, batch_size=int(1e5)):
    try:
        offset = 0
        all_injections = []
        all_detections = []

        while True:
            # Fetch a batch of UUIDs for injections matching the program
            uuids_batch = await client.frb_master.injections.find(
                {"injection_program": program_name}, {"_id": 0, "id": 1}
            ).skip(offset).limit(batch_size).to_list(None)

            if not uuids_batch:
                break  # No more records

            uuids = [uuid["id"] for uuid in uuids_batch]

            # Fetch corresponding injections
            injections = await client.frb_master.injections.find(
                {"id": {"$in": uuids}}, {"_id": 0}
            ).to_list(None)

            # Fetch detections, keeping only the highest SNR per det_id
            detections = await client.frb_master.detections.aggregate(
                [
                    {"$match": {"det_id": {"$in": uuids}}},
                    {"$sort": {"combined_snr": -1}},  # Sort by SNR descending
                    {
                        "$group": {
                            "_id": "$det_id",
                            "document": {"$first": "$$ROOT"},  # Keep first (highest SNR)
                        }
                    },
                    {"$replaceRoot": {"newRoot": "$document"}},
                    {"$project": {"_id": 0}},
                ],
                allowDiskUse=True
            ).to_list(None)

            # Merge batches into global lists
            all_injections.extend(injections)
            all_detections.extend(detections)

            offset += batch_size
            print(f"Processed batch {offset // batch_size}...")

        # Write final combined JSON
        merged_data = {"injections": all_injections, "detections": all_detections}
        with open(file_path, "w") as json_file:
            json.dump(merged_data, json_file, indent=4)

        print(f"✅ Merged data saved to: {file_path}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

async def main():
    client = AsyncIOMotorClient(
        host="localhost",
        port=27017,
        username="",
        password="",
    )
    program_name = "september_2025_catalog2_revertedRFIsifter"
    file_path = "./injections_data.json"

    await batch_fetch_and_save_injections(client, program_name, file_path)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
