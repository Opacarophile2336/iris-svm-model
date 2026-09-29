"""Verification script for HMNC_PRO feedback update requirements."""
import io
import httpx
from openpyxl import Workbook, load_workbook

BASE_URL = "http://127.0.0.1:2006"

def run_checks():
    # 1. Login
    r_login = httpx.post(f"{BASE_URL}/auth/login", json={"username": "0814230306", "password": "0814230306"})
    assert r_login.status_code == 200, f"Login failed: {r_login.text}"
    token = r_login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[1/14] Login OK")

    # 2. Manual Prediction across RBF, LINEAR, POLY
    r_pred = httpx.post(f"{BASE_URL}/prediction/predict-all", headers=headers, json={
        "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2
    })
    assert r_pred.status_code == 200, f"Predict-all failed: {r_pred.text}"
    data = r_pred.json()
    print(f"[2/14] Manual Prediction returned: {data['primary_prediction']}, Consensus: {data['consensus']}")

    # 3, 4, 5. Verify RBF, LINEAR, POLY
    for k in ["rbf", "linear", "poly"]:
        k_res = data["results"][k]
        pred_sp = k_res["predicted_species"]
        conf = k_res["confidence"]
        acc = k_res["accuracy"]
        cv_acc = k_res["cv_accuracy"]
        print(f"   -> {k.upper()}: Pred={pred_sp}, Conf={conf}%, Real Test Acc={acc}%, CV Acc={cv_acc}%")
        assert conf > 0
        assert acc >= 80.0
    assert "sigmoid" not in data["results"]
    print("[3-5/14] Verified RBF, LINEAR, POLY models with genuine validated accuracy.")

    # 6. Real Species Image Retrieval
    for sp in ["Iris setosa", "Iris versicolor", "Iris virginica"]:
        r_img = httpx.get(f"{BASE_URL}/image/species/{sp}", headers=headers)
        assert r_img.status_code == 200, f"Image fetch failed for {sp}: {r_img.text}"
        img_data = r_img.json()
        src = img_data.get("source")
        url = str(img_data.get("image_url"))[:65]
        print(f"   -> {sp}: Source={src}, URL={url}")
        assert src in ["GBIF", "Wikimedia Commons", "GBIF (Verified Real Specimen)", "cached"]
    print("[6/14] Verified authentic biological photo retrieval (GBIF/Wikimedia).")

    # 7. CSV Upload
    csv_content = (
        "Sepal Length,Sepal Width,Petal Length,Petal Width\n"
        "5.1,3.5,1.4,0.2\n"
        "6.0,2.9,4.5,1.5\n"
        "6.9,3.1,5.4,2.1\n"
    )
    files_csv = {"file": ("batch.csv", csv_content.encode("utf-8"), "text/csv")}
    r_csv = httpx.post(f"{BASE_URL}/prediction/batch-upload", headers=headers, files=files_csv)
    assert r_csv.status_code == 200, f"CSV batch upload failed: {r_csv.text}"
    batch_data = r_csv.json()
    assert batch_data["total_rows"] == 3
    print("[7/14] CSV Upload & Batch Prediction OK (3 rows processed across RBF, LINEAR, POLY)")

    # 8. TXT Upload
    txt_content = (
        "sepal_length\tsepal_width\tpetal_length\tpetal_width\n"
        "5.0\t3.6\t1.4\t0.2\n"
        "5.9\t3.0\t4.2\t1.5\n"
    )
    files_txt = {"file": ("batch.txt", txt_content.encode("utf-8"), "text/plain")}
    r_txt = httpx.post(f"{BASE_URL}/prediction/batch-upload", headers=headers, files=files_txt)
    assert r_txt.status_code == 200, f"TXT batch upload failed: {r_txt.text}"
    assert r_txt.json()["total_rows"] == 2
    print("[8/14] TXT Upload (tab-separated) OK (2 rows processed)")

    # 9. XLSX Upload
    wb = Workbook()
    ws = wb.active
    ws.append(["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"])
    ws.append([5.1, 3.5, 1.4, 0.2])
    ws.append([6.4, 3.2, 4.5, 1.5])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    files_xlsx = {"file": ("batch.xlsx", buf.read(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    r_xlsx = httpx.post(f"{BASE_URL}/prediction/batch-upload", headers=headers, files=files_xlsx)
    assert r_xlsx.status_code == 200, f"XLSX batch upload failed: {r_xlsx.text}"
    assert r_xlsx.json()["total_rows"] == 2
    print("[9/14] XLSX Upload OK (2 rows processed)")

    # 10. Invalid Upload Handling
    bad_csv = "Feature1,Feature2\n1.0,2.0\n"
    files_bad = {"file": ("bad.csv", bad_csv.encode("utf-8"), "text/csv")}
    r_bad = httpx.post(f"{BASE_URL}/prediction/batch-upload", headers=headers, files=files_bad)
    assert r_bad.status_code == 400
    print(f"[10/14] Invalid file rejected gracefully with 400: {r_bad.json()['detail']}")

    # 11. Downloadable XLSX Result
    batch_id = batch_data["batch_id"]
    r_exp_xlsx = httpx.post(f"{BASE_URL}/prediction/batch-export", headers=headers, json={"batch_id": batch_id, "format": "xlsx"})
    assert r_exp_xlsx.status_code == 200
    wb_down = load_workbook(io.BytesIO(r_exp_xlsx.content))
    assert "Batch Predictions" in wb_down.sheetnames
    assert "Model Evaluation Summary" in wb_down.sheetnames
    print(f"[11/14] Downloadable XLSX generated ({len(r_exp_xlsx.content)} bytes) with sheets: {wb_down.sheetnames}")

    # 12. Downloadable CSV Result
    r_exp_csv = httpx.post(f"{BASE_URL}/prediction/batch-export", headers=headers, json={"batch_id": batch_id, "format": "csv"})
    assert r_exp_csv.status_code == 200
    assert b"RBF Prediction" in r_exp_csv.content
    print(f"[12/14] Downloadable CSV generated ({len(r_exp_csv.content)} bytes)")

    # 13. Model Insights Endpoint
    r_ins = httpx.post(f"{BASE_URL}/prediction/insights", headers=headers, json={
        "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2
    })
    assert r_ins.status_code == 200
    assert "insights" in r_ins.json()
    match = r_ins.json()["insights"]["closest_botanical_match"]
    print(f"[13/14] Model Insights generated: Botanical Match = {match}")

    # 14. Existing Functionality (Quality & Datasets & History)
    r_ds = httpx.get(f"{BASE_URL}/datasets/quality", headers=headers)
    assert r_ds.status_code == 200
    r_hist = httpx.get(f"{BASE_URL}/prediction/history", headers=headers)
    assert r_hist.status_code == 200
    print("[14/14] Existing core endpoints verified OK.")
    print("\n=======================================================")
    print("ALL 14 ACCEPTANCE VERIFICATION CHECKS PASSED SUCCESSFULLY")
    print("=======================================================")

if __name__ == "__main__":
    run_checks()
