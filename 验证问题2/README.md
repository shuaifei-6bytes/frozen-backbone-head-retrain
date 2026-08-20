# 楠岃瘉闂 2锛欳lient 鍒犻櫎鐨勮繃搴﹂仐蹇?
杩愯鍏ュ彛涓?`run_experiment.py`銆傚畠瀹炵幇璁捐鍗曚腑鐨勪笁鏉′弗鏍奸厤瀵硅缁冭建杩癸細`M_full`銆乣M_A_neutral` 涓?`M_minus_A`锛屽苟淇濆瓨姣忎釜 seed 鐨勫垵濮嬫潈閲嶃€佸悇妯″瀷鏉冮噸銆佽缁冩竻鍗曘€佹寚鏍囧拰鏈€缁堝潎鍊?鏍囧噯宸眹鎬汇€?
```bash
python run_experiment.py --device cuda --data-dir /path/to/waterbirds --output-dir outputs --seeds 42 123 --global-rounds 15
```

鏍囧噯 Waterbirds 浠呮彁渚涘悎鎴愬悗鐨勫崟寮犲浘鍍忓強鍏惰儗鏅爣绛撅紝鏈彁渚涘悓涓€楦熶富浣撳湪鍙︿竴鑳屾櫙涓嬬殑娓叉煋銆傚洜姝や唬鐮佷笉浼氭妸鈥滃悓涓讳綋鑳屾櫙鏇挎崲鈥濅吉绉颁负宸插疄鐜帮細姣忎釜 seed 閮戒細鍐欏嚭 `counterfactual_scope.json`銆侰lient A 鐨勪袱涓缁冩潯浠跺浐瀹氬叾鏉ユ簮姹犮€佹牱鏈噺鍜屾爣绛炬暟锛屼絾閫氳繃鎸夌粍閲嶉噰鏍峰疄鐜?95/5 涓?25/25/25/25 鍒嗗竷锛涘叧绯荤炕杞寚鏍囦娇鐢ㄥ悓鏍囩銆佷笉鍚屽浘鍍忕殑鑳屾櫙瀵癸紝鏄繎浼煎弽浜嬪疄鎸囨爣銆?
