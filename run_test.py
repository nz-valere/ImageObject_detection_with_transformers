from TableDetector import TableDetector

detector = TableDetector()
results = detector.predict_from_path("sample_img.jpg", threshold=0.01)

if results:
    for r in results:
        print(f"Label: {r['label']}, Score: {r['score']}, Box: {r['box']}")
else:
    print("No tables detected.")
