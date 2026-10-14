# isaac_nav

連接 Isaac Sim 無人機(Pegasus ROS2 backend)與 NavRL ROS2 堆疊
(`map_manager`、`onboard_detector`、`navigation_runner`)的膠水層。ROS2 Jazzy、Isaac Sim 6.0.1。

NavRL 本身的 topic / service 介面沒有改動。這個套件只補上 NavRL 在 Unitree-Go2 上
由 Go2 驅動程式提供的東西:`Odometry` topic、同一個時鐘域的深度影像、
從 `cmd_vel` 到飛控的路徑、Isaac 專用的參數,以及一個 launch 檔。

```
isaac_nav/
├── package.xml  setup.py  setup.cfg  resource/isaac_nav     ament_python
├── isaac_nav/
│   ├── odom_bridge_node.py          state/pose + state/twist -> nav_msgs/Odometry
│   ├── depth_restamp_node.py        深度 Image,header.stamp := 節點時鐘
│   ├── cmd_vel_ardupilot_bridge.py  Twist -> SET_POSITION_TARGET_LOCAL_NED(ArduPilot SITL)
│   └── cmd_vel_px4_bridge.py        Twist -> offboard setpoint,解鎖 + OFFBOARD(PX4 SITL)
├── config/                          Pegasus Iris 用的 NavRL 參數
│   ├── map_param.yaml  dynamic_detector_param.yaml  yolo_detector_param.yaml
│   └── safe_action_param.yaml  navigation_param.yaml
└── launch/isaac_navrl.launch.py     啟動全部節點
```

## 建置

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install ros-jazzy-vision-msgs        # onboard_detector 需要
cd <ros2_ws>/src && cp -r /path/to/NavRL/ros2/{map_manager,onboard_detector,navigation_runner,isaac_nav} .
cd .. && colcon build --symlink-install       # 一定要 symlink-install,原因見下
source install/setup.bash
```

`--symlink-install` 不能省:`navigation_runner` 只安裝 `navigation_node.py` 和 `cfg/`;
`navigation_node.py` 會 import `navigation`、`ppo`、`utils`,並從自己所在的目錄載入
`ckpts/`,這些只有透過符號連結才找得到。

## 執行

```bash
# 1. 啟動 Isaac Sim + Pegasus(ROS2Backend pub_clock=True、320x240 含深度的相機)與 SITL
# 2. NavRL
ros2 launch isaac_nav isaac_navrl.launch.py \
    backend:=px4 \
    navrl_site_packages:=/path/to/venv/lib/python3.12/site-packages
# 3. 起飛後送出目標
ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
    "{header: {frame_id: map}, pose: {position: {x: 5.0, y: 2.0, z: 0.0}}}"
```

常用參數:`drone_id`、`backend:=px4|ardupilot|none`(預設 `px4`)、`use_sim_time`、
`use_yolo`、`odom_topic`、`cmd_vel_topic`、`twist_topic`、`mavlink_connection`、
`pymavlink_site_packages`、`navrl_site_packages`、`vel_limit`,相機
(`fx fy cx cy image_cols image_rows camera_x camera_y camera_z`)、地圖
(`map_size_x/y/z ground_height`)、安全層(`min_height max_height`)、
`enable_foxglove`。

## Topic 對照表(drone_id 0)

| Isaac Sim(Pegasus) | isaac_nav | NavRL |
|---|---|---|
| `/drone0/state/pose` PoseStamped、BEST_EFFORT、ENU | `odom_bridge_node` 輸入 | |
| `/drone0/state/twist` TwistStamped、BEST_EFFORT、FLU 機體座標 | `odom_bridge_node` 輸入 | |
| | `/drone0/odom` Odometry,frame `map`,機體座標 twist | `navigation_node`、`map_manager_node`、`dynamic_detector_node` 的 `odom_topic` |
| `/drone0/camera/depth` Image 32FC1(m) | `depth_restamp_node` 輸入 | |
| | `/drone0/camera/depth_restamped` | `map_manager_node`、`dynamic_detector_node` 的 `depth_image_topic` |
| `/drone0/camera/color/image_raw` | | `dynamic_detector_node`、`yolo_detector_node` 的 `color_image_topic` |
| `/drone0/camera/color/camera_info` | 預設內參的來源(不訂閱) | |
| `/clock` | `use_sim_time` | 所有節點 |
| | `/drone0/cmd_vel` Twist(FLU 機體座標) | 由 `navigation_node` 發布(`cmd_topic`),cmd_vel bridge 接收 |
| 連到 SITL 的 MAVLink | cmd_vel bridge(ArduPilot `udpout:127.0.0.1:14551+10*id`,PX4 `udpin:0.0.0.0:14540+10*id`) | |
| | | `/goal_pose` PoseStamped 輸入、`/navigation_emergency_stop` Bool 輸入(未改動) |
| | | service `/occupancy_map/raycast`、`/onboard_detector/get_dynamic_obstacles`、`/safe_action/get_safe_action`(未改動) |

NavRL 的節點刻意不加 namespace:NavRL 原始碼裡的 service 與目標 topic 名稱是相對名稱,
加了 namespace 會被改名。

## 已用真實 Pegasus + PX4 SITL 驗證的項目

* **相機安裝方向。** Pegasus 相機朝正前方、正立、沒有鏡像:對正前方 2.47 m 的平牆,NavRL 自己的
  `/occupancy_map/raycast` 在 0 度回傳 2.5 m,在 ±10/20/30 度回傳對稱的距離(2.5、2.7、2.9),
  符合 2.47/cos(theta)。所以 `camera_x = 0.30`、`body_to_depth_sensor` 裡的光學座標到 FLU 的旋轉,
  以及 `camera_info` 來的內參都是正確的。
* **深度。** 320x240 `32FC1`,單位公尺,約 26 Hz;在高度 2.2–2.6 m、`ground_height: 0.15` 時,
  地面不會出現在地圖裡。
* **閉迴路。** `backend:=px4`,飛機在約 2.2–2.5 m 懸停:bridge 自己切到 OFFBOARD
  (飛機當時已解鎖)。目標在側面 4 m:NavRL 轉向目標、以 1.0 m/s 飛行並停在距離目標 1.0 m 處
  (這是它的設計),高度偏差 0.04 m。目標在牆後:沒有碰撞(最近距離 x 方向 4.0 m,牆面約 5.5 m),
  高度偏差 0.08 m;因為策略完全是局部的,它會在牆前徘徊。
* **時鐘。** Pegasus 用牆上時間(約 1.79e9 s)標記 `state/*`,但 `/clock` 是模擬時間(約 3000 s);
  bridge 會用節點時鐘重新標記時間,`use_sim_time:=true` 時就是模擬時鐘。模擬跑在大約 0.4 倍
  實時,所以 NavRL 的 20 Hz 計時器在牆上時間只有約 9 Hz。

## 尚未驗證的假設

1. **機體座標 twist。** `state/twist` 文件寫的是 FLU 機體速度;沒有在機頭轉向的狀態下和
   `state/twist_inertial` 比對(上面的飛行只能間接支持)。
2. **安全層高度。** `min_height 0.5` / `max_height 5.0` 只是框住巡航高度,沒有在接近上下限時測試過。
3. **ArduPilot backend。** 只飛過 PX4。ArduPilot bridge 及其 14551 埠(依參考 launch 檔的註解推測的
   companion link)沒有執行過。
4. **移動障礙物。** `dynamic_detector` 在靜態場景回報 0 個障礙物;移動障礙物的偵測沒有測試,
   `use_yolo` 也沒有。
5. **跟影像大小有關的濾波。** `depth_filter_margin` 為了 320x240 減半成 5。
6. **PX4 bridge 自己解鎖並進入 OFFBOARD**(`auto_arm_offboard` 在參考 bridge 預設為 true,維持不變),
   起飛前也一樣。
7. **偏航對齊。** 參考的角度控制器把未包裹的原始角度差直接丟給 PID,所以可能繞遠路轉(觀察過一次:
   飛出去前先轉了約 180 度)。
8. **Python 環境。** `navigation_node.py` 只需要 `torch`、`hydra-core`/`omegaconf` 和 `numpy`
   (不再使用 torchrl/tensordict/einops)。把 `navrl_site_packages` 指到有這些套件的
   site-packages 目錄;Isaac Lab 的 venv 可以用。

## 對 NavRL 原始碼做的修改(Jazzy 上、沒有舊版 torchrl 時需要)

* `map_manager/.../occupancyMap.h`、`onboard_detector/.../dynamicDetector.h`:`cv_bridge/cv_bridge.h`
  -> `cv_bridge/cv_bridge.hpp`(Jazzy 只提供 `.hpp`)。
* `onboard_detector/.../dynamicDetector.cpp`:`visCB()` 在第一次 `detectionCB()` 建立 `uvDetector_` 之前
  直接 return;不加的話,啟動時深度影像還沒到,節點會 segfault。
* `navigation_runner/scripts/navigation.py`:啟動時也等待 `/onboard_detector/get_dynamic_obstacles`。
  原本只等 raycast service;在另一個 server 被發現前做的同步 `call()` 會遺失且永遠不返回,
  只要節點同時啟動,控制迴圈就會卡死(完全沒有 `cmd_vel`)。
* `navigation_runner/scripts/ppo.py`:改用純 PyTorch 的推論策略,取代 torchrl/tensordict 版本。
  state-dict 的 key 沒變,所以 `ckpts/navrl_checkpoint.pt` 可以用 `strict=True` 載入。
* `navigation_runner/scripts/navigation.py`、`utils.py`:不再 import torchrl/tensordict
  (spec 和 `TensorDict` 觀測改成普通 dict;只在訓練用的 `make_batch` 移除)。

## 其他測試

* 新策略與原本 torchrl 0.6 程式碼在官方 checkpoint 上、40 筆隨機觀測的比較:
  torch 2.5.1 下輸出逐位元相同;torch 2.11 不用 torchrl 時最大差異 1.5e-6。
* 用合成 publisher 測 `odom_bridge_node` / `depth_restamp_node`;在 Jazzy 上 `colcon build` 四個套件;
  `backend:=px4|ardupilot|none`(預設 `px4`)的 launch 參數處理。

## 注意事項

* 如果 Pegasus 沒有發布 `/clock`,要用 `use_sim_time:=false`,否則計時器永遠不會觸發。
* 不要同時執行 MAVROS 和 `cmd_vel_px4_bridge`:兩者都會綁 UDP 14540。
* 停止 launch 之後,檢查有沒有殘留的 `navigation_node.py`(它卡在 service 呼叫時會忽略 SIGTERM)。

## 隨機目標(goal_manager_node + Isaac Sim 顯示)

`goal_manager_node`(ROS2)隨機取樣 N 個 3D 目標,一次一個發布到 `/goal_pose`,
等無人機在 `reach_radius` 內停留 `dwell_s` 後換下一個(或經過 `goal_timeout_s` 後標記為失敗)。
Isaac Sim 只負責畫出目標:`isaac_scripts/goal_display.py` 會由 Pegasus 的
`multi_backend_example` 擴充功能自動載入(容器內路徑 `/Volume/NavRL/...`,可用環境變數
`NAVRL_GOAL_DISPLAY` 覆寫;修改擴充功能後要重啟 Isaac Sim)。
它監聽 UDP 9870,把每個目標畫成沒有碰撞的彩色 USD 球
(灰色待處理、青色下一個、黃色目前這個並放大 1.6 倍、綠色已到達、紅色失敗)。
半徑是 goal manager 的 `marker_radius`(預設 0.25 m)。這些球是真的網格,所以無人機的
深度相機看得到,NavRL 會把它們當成障礙:會繞開球而不是飛進去。半徑要保持小
(目標進入 `reach_radius` 內就算到達,見下)。同樣的狀態也以
`/goal_manager/markers`(MarkerArray)發布給 Foxglove / RViz。

- 障礙物與可到達性的過濾需要靜態格子。`goal_display.py` 每載入一個場景(無人機生成、模擬播放中)
  會自己匯出一次,x/y 範圍取自場景碰撞體(排除無人機、目標標記、地面,並限制在無人機周圍
  ±40 m),z 為 0..5 m,輸出到 `~/Volume/NavRL/maps/scene_grid.npz`。
  該檔案存在時,`goal_manager.launch.py` 會把它當作 `map_file`。目標會從與無人機起點格相連的
  空格中挑選。沒有地圖時只能使用 NavRL 的即時地圖(相機看過的部分)。要手動重做或改範圍:
  `ros2 run isaac_nav grid_export_request --x -20 20 --y -20 20 --z 0 5`。
- `reach_radius` 必須大於 NavRL 的 `goal_stop_distance`(`isaac_navrl.launch.py` 預設 0.5 m):
  NavRL 進到這個距離內就不再下速度指令,無人機會懸停在約 0.5 m 處,實測停在 0.50–0.52 m。
  `reach_radius` 不比它大的話,懸停位置差 1–2 cm 就不算到達,目標會一直拖到逾時。
  節點內建預設是 0.8 m,但 `goal_manager.launch.py` 的預設是 0.5 m(目前剛好在邊界上);
  建議用 `reach_radius:=0.8`(至少比 `goal_stop_distance` 大 0.3 m)。距離用 3D 計算
  (`use_goal_height:=true` 時含高度差)。
- `use_goal_height` 必須與 navigation_node 的參數一致(只有此時才檢查 z)。
- 重新產生目標:`ros2 topic pub --once /goal_manager/regenerate std_msgs/msg/Empty "{}"`。
- 測試狀況:
  * 目標邏輯用合成 odometry 和格子測過(300 個 seed 下沒有目標落在牆裡或封閉房間)。
  * 顯示與格子匯出用玩具場景在無頭 Isaac Sim 測過。
  * 完整閉迴路用 `12_multi_backend_standalone.py`(無頭 Isaac Sim、Pegasus Iris、PX4 SITL,
    預設場景索引 9)跑過 `seed:=0`、10 個目標:靜態地圖規劃 10/10 到達;即時地圖規劃修正後 10/10 到達;
    側向繞路(`use_detour`)6/10 到達(其中 4 個逾時)。這些都是各跑一次,沒有重複次數,
    也沒有對照組,只能說明可行,不能當作成功率。
  * 碰撞監控(用 odometry 的速度跳動與傾角判斷)在最後一次完整執行只有起飛瞬間一筆;
    更早的一次即時規劃執行曾在第 7 個目標前翻機(PX4 `Attitude failure (roll)`),原因沒查出來,之後沒再重現。
  * **未**逐項驗證:Pegasus 擴充功能的 Spawn 按鈕流程、串流版 Isaac Sim 裡球的顯示大小。

### 一鍵啟動(`navrl_stack.launch.py`)

啟動 PX4(`~/run_px4.sh`)、`isaac_navrl.launch.py`,5 秒後再啟動 `goal_manager.launch.py`
(Isaac Sim 本身另外啟動:擴充功能,或
`PegasusSimulator/examples/12_multi_backend_standalone.py`)。命令列上每個 `name:=value`
都會轉給兩個 launch 檔,預設值重現平常的兩條指令(`use_goal_height:=true vel_limit:=1.0
goal_ignore_radius:=0.3 num_goals:=10 marker_radius:=0.01 ...`)。控制參數:`run_px4`、
`px4_script`、`navrl`、`goal_manager`、`goal_manager_delay`、`px4_kill_pattern`。
在 Isaac Sim 重新載入場景後只需要重啟 PX4:`navrl:=false goal_manager:=false`。
Ctrl-C 也會把 PX4 關掉。

#### 不開 Isaac Sim 視窗執行(standalone)

需要兩個終端機。Isaac Sim 在容器內無頭執行;每次重開 Isaac Sim,PX4 也必須重開,
所以先啟動 ROS 端,幾秒後再啟動 Isaac Sim。

```bash
# 1. PX4 + NavRL + goal_manager(主機)
cd ~/Volume/NavRL/ros2 && source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch isaac_nav navrl_stack.launch.py planner_source:=live enable_foxglove:=false

# 2. Isaac Sim,無 GUI(在主機下指令,實際在容器內執行)
docker exec -u root -e HOME=/root -e ROS_DISTRO=jazzy \
  -e LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/root/.ros/fastdds.xml isim-peterson-isaac-sim-1 \
  /isaac-sim/python.sh /Volume/PegasusSimulator/examples/12_multi_backend_standalone.py --headless
```

* 需要 `-u root` 和那幾個環境變數,否則容器內的 ROS 2 topic 到不了主機。
* 拿掉 `planner_source:=live` 就會在匯出的 `map_file` 上規劃(靜態);拿掉
  `enable_foxglove:=false` 就會在 8765 埠開 Foxglove。
* 在第一個終端機按 Ctrl-C 停止(會一併關掉 PX4)。殘留的 Isaac Sim 程序:
  `docker exec -u root isim-peterson-isaac-sim-1 pkill -f 12_multi_backend_standalone`。
* 不要在同一個容器的串流版 Isaac Sim 還連著同一個 PX4 時同時執行:兩個模擬器會搶同一個 PX4,
  也會共用 GPU。

### 忽略 NavRL 光達裡的目標標記(`goal_ignore_radius`,預設 0 = 關閉)

Isaac 的目標球是網格,所以深度相機會把它們畫進地圖當成障礙,NavRL 可能在目標前停住。
goal_manager 會把所有標記位置發布到 `/goal_manager/goal_points`(PoseArray、transient local);
使用 `isaac_navrl.launch.py goal_ignore_radius:=0.3` 時,NavRL 會把距離任何標記在該半徑內的
光達命中點當成「沒打到」(最大距離)。副作用:標記半徑內真正的障礙也會被忽略,
而那條射線上標記後面的東西也看不到。

### 側向繞路(goal_manager_node、`use_detour`,預設關閉,不需要地圖)

用於大型障礙擋在無人機與目標之間、NavRL 一直往裡推的情況。如果在 `stuck_window_s`(8 秒)內,
到目標的距離縮短不到 `stuck_progress`(0.5 m),節點會拍一張光達快照(`/occupancy_map/raycast`,
和 NavRL 同樣的 36 x 4 條光束),並送給 NavRL 一個臨時中繼點:在無人機到目標連線較空的一側、
距離 `detour_offset`(2 m)(稍微偏前)。到達中繼點(`detour_reach`)或經過 `detour_timeout_s` 後
回到目標;如果又卡住,就在同樣距離試另一側,接著兩側都再放大 `detour_growth` 倍
(較空的一側、另一側、4 m 較空側、4 m 另一側、8 m、16 m …… 直到 `detour_max`,預設共 8 次),
全部失敗才把目標標記為失敗。光達快照只用來決定先走哪一側;每個中繼點都以完整距離送出,
由 NavRL 自己避開看得到的東西。無人機朝某個中繼點在 `stuck_window_s` 內沒有進展就放棄它
(「blocked」),回到目標。與 `use_planner` 互斥(兩者都開時節點拒絕啟動):
`ros2 launch isaac_nav goal_manager.launch.py use_planner:=false use_detour:=true`。
第一次起飛結束之前不檢查:高度 >= `detour_min_altitude` 且 1 秒內沒有垂直移動
(只判斷一次,之後不再看),或在距離目標 reach_radius + 0.5 m 內時也不檢查。

### 全域路徑規劃(goal_manager_node、`use_planner`,預設開啟)

有 `map_file` 時,goal manager 會在匯出的格子上做 A* 規劃(距離變換代價讓路徑遠離膨脹邊界,
再做視線平滑),並把沿路徑往前 `lookahead` 公尺的移動 carrot 點(以 5 Hz 發到 `/goal_pose`)
送給 NavRL,取代遠方的目標。carrot 會被拉回視線範圍內最遠的點,所以不會跑到轉角後面。
路徑發布在 `/goal_manager/path`。當無人機偏離路徑超過 `replan_distance`,就從當前位置重新規劃。
到達判斷仍然使用最終目標。沒有路徑時直接飛向目標,並每 2 秒重試規劃(無人機可能從格子外面起飛);
沒有 `map_file` 時一律直飛。`goal_list:="x,y,z;x,y,z"` 可以用固定目標取代隨機目標。
`use_goal_height:=false` 時路徑保持在無人機當前的 z 層(此時 NavRL 會忽略目標高度)。

已測試:合成牆(繞過牆端、封閉目標 -> 無路徑、跟隨 carrot 的模擬無人機到達兩個目標),
以及上面「隨機目標」一節列出的 Isaac Sim standalone 完整執行(靜態地圖規劃 10/10)。
測試要在獨立的 ROS domain 執行(`ROS_DOMAIN_ID=88 ROS_LOCALHOST_ONLY=1`),
因為節點會發布 /goal_pose、測試會發布 /drone0/odom,會干擾正在運作的系統。

#### 即時地圖規劃(`planner_source:=live`)

規劃器和 carrot 跟隨都相同,但格子不是匯出的 `map_file`:而是飛行中從 NavRL 自己的佔據地圖
(`/occupancy_map/voxel_map`,約 20 Hz、約 50 kB)建出來。未知空間視為空,所以路徑會直走到
看見障礙為止,再(`live_replan_s`,1 秒)繞開重新規劃。map_manager 只發布無人機周圍的一個方塊
(`live_window`,總共 8 x 8 x 3 m,即 x/y ±4 m、往上 ±1.5 m),所以節點自己維護一張持久的格子
(`live_resolution` 0.5 m、`live_origin`/`live_size` 60 x 60 x 6 m):方塊內以快照取代格子內容,
方塊外保留先前的知識。這份記憶讓無人機不會再走進已經看過的死路。障礙物會膨脹
`live_inflate`(0.6 m),和匯出的格子一樣。
`map_file` 此時只用來挑目標(可省略)。`goal_ignore_radius`(與 NavRL 的是同一個參數,
由 `navrl_stack.launch.py` 同時傳給兩邊)會丟掉目標標記周圍的地圖點,那些點是深度相機看到的球;
最後 `lookahead` 公尺直接飛向目標。

    ros2 launch isaac_nav navrl_stack.launch.py planner_source:=live
