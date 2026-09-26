"""What actually happens in this video, second by second. Raw frames, no overlay."""
import numpy as np, cv2, probe_common
d = probe_common.load()
fi = d["frame_index"]; t = d["times"]; col = d["colour"]
picks = [int(np.argmin(np.abs(t - s))) for s in np.arange(0, t[-1]+0.01, 1.0)]
tiles = []
for k in picks:
    a = fi[k]
    if a not in col: continue
    img = cv2.resize(col[a], None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(img, (0,0), (img.shape[1],18), (0,0,0), -1)
    cv2.putText(img, f"t={t[k]:.1f}s", (5,13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
    tiles.append(img)
cols = 5
rows = [np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles)-cols+1, cols)]
sheet = np.vstack(rows)
sheet = cv2.resize(sheet, (1700, int(1700*sheet.shape[0]/sheet.shape[1])))
cv2.imwrite("story.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(len(tiles), "tiles", sheet.shape)
