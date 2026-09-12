import json, sys, numpy as np, cv2
d = sys.argv[1]
idx = json.load(open(f"{d}/index.json"))
for k, v in idx.items():
    im = cv2.imread(f"{d}/{k}.png", cv2.IMREAD_UNCHANGED)
    h = im.shape[0]
    t = im[int(h*0.28):int(h*0.55)]
    m = t[..., 3] > 128
    if m.sum() < 10: v["kit"] = "#888888"; v["shorts"] = "#333333"; continue
    px = t[m][:, :3].astype(float)
    # pick most saturated cluster-ish: median of top-40% saturation pixels
    hsv = cv2.cvtColor(px.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1,3)
    sel = px[hsv[:,1] >= np.percentile(hsv[:,1], 60)]
    b, g, r = np.median(sel, axis=0)
    v["kit"] = "#%02x%02x%02x" % (int(r), int(g), int(b))
    s = im[int(h*0.55):int(h*0.72)]; ms = s[..., 3] > 128
    if ms.sum() >= 10:
        b, g, r = np.median(s[ms][:, :3], axis=0); v["shorts"] = "#%02x%02x%02x" % (int(r), int(g), int(b))
    else: v["shorts"] = "#333333"
json.dump(idx, open(f"{d}/index.json", "w"))
print(list(idx.items())[:5])
