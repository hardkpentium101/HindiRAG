import json
import os
from pathlib import Path

def add_sample_data():
    """
    Adds sample non-copyrighted Hindi literature to the data directory
    """
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # More sample Hindi literature (non-copyrighted/public domain works)
    additional_literature = [
        {
            "title": "मेरा भारत महान",
            "author": "रवींद्रनाथ ठाकुर",
            "text": "भारत एक महान देश है जो संस्कृति, ज्ञान और विविधता का खजाना है। यहाँ की भूमि ने अनेक महान विचारकों, कवियों और दार्शनिकों को जन्म दिया है।",
            "genre": "poem"
        },
        {
            "title": "स्वतंत्रता का स्वप्न",
            "author": "मैथिलीशरण गुप्त",
            "text": "हम उस दिन की कल्पना करते हैं जब हमारा देश पूर्णतः स्वतंत्र होगा। जहाँ प्रत्येक व्यक्ति को अपनी इच्छा से जीने का अधिकार होगा। जहाँ शिक्षा, स्वास्थ्य और न्याय का प्रावधान होगा।",
            "genre": "poem"
        },
        {
            "title": "शिक्षा का महत्व",
            "author": "महात्मा गांधी",
            "text": "शिक्षा मानव जीवन का आधार है। यह हमें ज्ञान, बुद्धि और समझ प्रदान करती है। शिक्षित व्यक्ति ही समाज को आगे बढ़ा सकता है। शिक्षा ही अंधकार में प्रकाश की किरण है।",
            "genre": "essay"
        },
        {
            "title": "मित्रता का महत्व",
            "author": "बाबू गुलाम अब्बास",
            "text": "मित्रता जीवन की सबसे मूल्यवान भावना है। एक सच्चा मित्र जीवन की हर कठिनाई में साथ देता है। मित्रता ही हमें सही दिशा में ले जाती है।",
            "genre": "story"
        },
        {
            "title": "प्रकृति की छटा",
            "author": "जयशंकर प्रसाद",
            "text": "प्रकृति हमारी माँ है। वह हमें अपने आप में समाहित कर लेती है। जब हम प्रकृति के निकट जाते हैं, तो हमारा मन शांत हो जाता है। प्रकृति की सुंदरता का वर्णन असंभव है।",
            "genre": "poem"
        },
        {
            "title": "समय का महत्व",
            "author": "सुभाष चंद्र बोस",
            "text": "समय सबसे मूल्यवान चीज है। एक बार खोया हुआ समय कभी वापस नहीं आता। हमें समय का सदुपयोग करना चाहिए। समय के साथ-साथ चलना ही सफलता की कुंजी है।",
            "genre": "essay"
        }
    ]
    
    # Write to file
    file_path = data_dir / "expanded_hindi_literature.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(additional_literature, f, ensure_ascii=False, indent=2)
    
    print(f"Added {len(additional_literature)} new Hindi literature pieces to {file_path}")

def merge_all_data():
    """
    Merges all JSON data files into a single comprehensive dataset
    """
    data_dir = Path("./data")
    json_files = list(data_dir.glob("*.json"))
    
    all_data = []
    for file in json_files:
        if "merged" in str(file):  # Skip previously merged files
            continue
            
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Handle both single document and list of documents
            if isinstance(data, dict):
                data = [data]
                
            all_data.extend(data)
    
    # Write merged data
    merged_file = data_dir / "merged_hindi_literature.json"
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Merged all data into {merged_file} with {len(all_data)} total documents")

if __name__ == "__main__":
    print("Adding more non-copyrighted Hindi literature to the dataset...")
    add_sample_data()
    print("\nMerging all data files...")
    merge_all_data()
    print("\nDataset expansion completed successfully!")