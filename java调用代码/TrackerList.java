package detectmotion;
import com.google.gson.Gson;
import detectmotion.interestarea.IOTTransform;
import detectmotion.interestarea.PerspectiveConversion;
import detectmotion.tuple.Tuple2;
import detectmotion.tuple.Tuple3;
import detectmotion.utils.PHASE;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.log4j.Logger;
import org.apache.spark.sql.sources.In;
import org.opencv.core.*;
import org.opencv.tracking.*;
import spark.type.VideoEventData;

import java.io.Serializable;
import  java.util.*;
/**
 * @author ：tyy
 * @date ：Created in 2020/5/1 16:24
 * @description： 创建一个保存已经识别的车辆信息的List，提供基本的增删改查，即涉及单个结点的管理维护，该类会修改TrackerList的值
 * @modified By：
 * @version: $
 */
@NoArgsConstructor
@Slf4j
public class TrackerList implements Serializable {
    private static final Logger logger = Logger.getLogger(TrackerList.class);
    @Getter  @Setter
    private  String  key  = null;//用来保存检测的对象是哪一个视频帧
    @Setter @Getter
    private String selectedType = "KCF";
    //对车辆进行维护，在增加一个tracker就相当于增加一辆车，与trackers完全同步
    @Setter @Getter
    private  List<CarDes> trackers = new LinkedList<>();
    @Setter @Getter
    private long startCount = 0;//计数

    public TrackerList(List<CarDes> trackers, long startCount,String selectedType) {
        this.trackers = trackers;
        this.startCount = startCount;
        this.selectedType  = selectedType;
    }

    Tracker createTrackerByName(String trackerType) {
        String[] trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};

        Tracker tracker = null;
        if (trackerType.equals( trackerTypes[0]))
            tracker = TrackerBoosting.create();
        else if (trackerType.equals(trackerTypes[1]))
            tracker = TrackerMIL.create();
        else if (trackerType.equals(trackerTypes[2]))
            tracker = TrackerKCF.create();
        else if (trackerType.equals(trackerTypes[3]))
            tracker = TrackerTLD.create();
        else if (trackerType.equals(trackerTypes[7]))
            tracker = TrackerCSRT.create();
        else {

            System.out.println("Incorrect tracker name:"+trackerType+"\n");
            System.out.println("Available trackers are: ");
            Integer str;
            for (int i = 0; i < trackerType.length(); i++) {
                System.out.print(trackerTypes[i] + " ");
            }
        }
        return tracker;
    }//可以灵活应对新增加或者修改的tracker (Factory,反摄）E

    public long getCurTrackersCount() {
        return startCount;
    }
    /***
     *
     * @param frame 更新的帧的信息
     * @param cars 要更新的位置
     * NOTE：会删除内部的所有tracker，所以调用的时候需要全部检查完成
     *           删除没有检测到的index
     */
    public void updateTrackPos (Mat frame,  Map<Integer,Rect2d> cars ){
        if(cars == null || cars.size() <= 0){
            return;
        }
        ArrayList<CarDes> needModify = new ArrayList<>();
        logger.info("tracker size ===" + trackers.size());
        cars.forEach((index,pos)->{
            CarDes car= trackers.get(index);
            logger.debug(key+": 原来预测到的位置" + car.getPos());
            logger.debug(key+": 现在检测到的位置" + pos);
            car.setPos(new Rect2d((car.getPos().x + pos.x) /2,
                    (car.getPos().y + pos.y) /2,pos.width,pos.height));
            needModify.add(car);
        } );
        logger.info(key + ": updateTrackpos: 检测到的重合对象，并且重新创建更新tracker" + needModify.size());
        updateTrackPos(frame,needModify);
//        logger.info(key + ": updateTrackpos：删除没有检测到重合的对象" + (trackers.size() - needModify.size()));
//        trackers.retainAll(needModify);

    }
    //更新tracker
    public void updateTrackPos (Mat frame,  List<CarDes> cars ){
        for(CarDes car: cars){
            Tracker t = createTrackerByName(this.selectedType);
            logger.info(key + " 更新重合 count = " + car + "的位置");
            t.init(frame,car.getPos());
            car.setTracker(t);
        }
    }

    //发现新的车辆
    public void createNewTrackersByArea(Mat frame,  List<Rect2d> cars ){
        logger.info(key + " create Cars" + cars.size());
        for(Rect2d carpos: cars){
            logger.info(carpos+";");
            Tracker t = createTrackerByName(selectedType);
            t.init(frame,carpos);
            CarDes c = new CarDes(carpos,startCount++,t,0);// 与上面的区别 是否创建新的车辆
            trackers.add(c);//对count进行设置

        }
    }
    public void setNewTracker(Mat frame,ArrayList<Integer> indexesofTracker ){
        for (int i = 0; i < indexesofTracker.size(); i++) {

           Integer index =  indexesofTracker.get(i);
           CarDes car= trackers.get(index);
           Tracker t = createTrackerByName(selectedType);
           t.init(frame,car.getPos());
           car.setTracker(t);
        }
    }

    //iotArea表示感兴趣的区域，如果更新之后不再感兴趣的区域内就删掉
    public void update(Mat frame){
        logger.warn(key + " enter update tracker");
        int length = trackers.size();
        for (int i = 0; i < length; i++) {
            trackers.get(i).setPhase(PHASE.TRACKER);
            //保存更新结果
            Rect2d newPos = new Rect2d();
            //目标没有丢失
            Tracker t = trackers.get(i).getTracker();
            if (t == null) {
                logger.info(key + ": " + trackers.get(i).getPos() + "trackers == null,count ==" + trackers.get(i).getCount());
                trackers.get(i).setMarkedDelete();
                return;
            }
            boolean res = t.update(frame, newPos);
            if (res != true) {
                trackers.get(i).setMarkedDelete();
                logger.info(key + " 丢失目标 count " + trackers.get(i) + "because update cannot see");
            } else {
                logger.warn(key + "更新之前" + trackers.get(i));
                trackers.get(i).setPos(newPos);
                logger.warn(key + "更新位置" + trackers.get(i));
            }
        }
        deleteOutFrame(frame.size());
        logger.warn(key + " leave update tracker");
        return  ;

    }
    //删除y方向上超过界限的边框
    public  void  deleteOutFrame(Size frameSize){
        int height = (int) frameSize.height;
        ArrayList<CarDes> deleted = new ArrayList<>();
        for (CarDes tracker : trackers) {
            Rect2d rect = tracker.getPos();
            double rh = ( (rect.y + rect.height) - height);
            if( rh > 0 ){
                deleted.add(tracker);//删除y轴超过了界限的
                log.info("删除y轴超过边界的边框" + rect);
                continue;
            }
        }
        trackers.removeAll(deleted);//删除deleted
    }

    public  void cleanLostedTrackers(int losttime){

       trackers.removeIf((carDes -> {
           if(carDes.markedLost >= losttime)
               logger.warn(key +" remove count = " + carDes.count +"because of track lost long time");
           return  carDes.markedLost > losttime;
       }));

    }

    public  void updatePhaseAndnexPoses(){
        for (int i = 0; i < trackers.size(); i++) {
            CarDes car = trackers.get(i);
            car.setPhase(PHASE.DETECTOR);
            car.setPreviousPos(car.getNetxPos());
            car.setNetxPos(car.getPos());

        }
    }

    public  void deletedNotInArea( PerspectiveConversion iot){
        List<CarDes> deleted = new LinkedList<>();
        //删除所有超过范围的追踪器
        if(iot != null){
            for (int i = 0; i < trackers.size(); i++) {
                Rect2d pos = trackers.get(i).getPos();
                if(!iot.isInsidePicArea(pos.br()) && !iot.isInsidePicArea(pos.tl()) ){
                    logger.warn(key +" remove count = " + trackers.get(i).count +"because   it goes beyond the interested  zone");
                    deleted.add(trackers.get(i));
                }
            }
        }
        trackers.removeAll(deleted);

    }

    public  TrackerList setTrackerType(String selectedType) {
        this.selectedType = selectedType;
        return  this;
    }

    //获取列表进行操作
    public  ArrayList<Rect2d> getTrackedCarsPos(){
        ArrayList< Rect2d> poses = new ArrayList<>();
        for (CarDes tracker : trackers) {

            poses.add(tracker.getPos());
        }
        return poses;

    }
    public  ArrayList<Tuple2<Rect2d,Long>> getPosAndCount(){
        ArrayList<Tuple2<Rect2d,Long>> arr = new ArrayList<>(trackers.size());
        for (int i = 0; i < trackers.size(); i++) {
            CarDes car = trackers.get(i);
            arr.add(new Tuple2<>(car.getPos(),car.getCount()));
        }
        return  arr;
    }
    public  ArrayList<Tuple3<Rect2d,Double,Double>> getSpeed(double time,PerspectiveConversion iot){
        ArrayList<Tuple3<Rect2d,Double,Double>> arr = new ArrayList<>(trackers.size());

        for (int i = 0; i < trackers.size(); i++) {
            CarDes car = trackers.get(i);
            Rect2d pre = car.getPreviousPos();
            Rect2d p = car.getNetxPos();
            if(pre == null || p == null)
                    continue;
            if(car.getPhase() == PHASE.TRACKER && car.speed != 0){//速度为0 不显示
                arr.add(new Tuple3<>(car.getPos(),car.speed,car.getCarLength()));
                continue;
            }

            if(car.getPhase() == PHASE.DETECTOR ) {
                //计算速度/////
                if (( car.speed != 0) &&
                        (!iot.isInsidePicArea(getRectCenter(pre)) || !iot.isInsidePicArea(getRectCenter(p)))) {
                    arr.add(new Tuple3<>(car.getPos(), car.speed,car.getCarLength()));
                } else if (iot.isInsidePicArea(getRectCenter(pre)) && iot.isInsidePicArea(getRectCenter(p))){
                    double s = calculateSpeed(p,pre,time,iot);
                    double carLength =  calculateCarLength(p.tl(),p.br(),iot);
                   // System.out.println(carLength);
                    car.setCarLength(carLength);
                    if (car.speed == 0) {
                        car.speed = s;
                    } else {
                        car.speed = (car.speed +  s) / 2;
                    }
                    arr.add(new Tuple3<>(car.getPos(), car.speed,car.getCarLength()));
                }
            }
        }
        return  arr;
    }

    private  Point getRectCenter(Rect2d r){
        return  new Point(r.x + r.width/2,r.y + r.height /2);
    }
    private double calculateCarLength(Point tl,Point br,PerspectiveConversion iot){
        double yratio = iot.getYRatio();
        List<Point> list = Arrays.asList(tl, br);
        List<Point> res = iot.transformPointList(list);
//        System.out.println(res.get(0));
//        System.out.println(res.get(1));
        double carLength = Math.sqrt(((res.get(0).y - res.get(1).y) * (res.get(0).y - res.get(1).y))*yratio*yratio);
 //       System.out.println(res.get(0).y - res.get(1).y);
        return  carLength;
    }
    private  double calculateSpeed(Rect2d pre,Rect2d p,double time,PerspectiveConversion iot){
        double xratio = iot.getXRatio();
        double yratio = iot.getYRatio();
        Point precenter = getRectCenter(pre);
        Point pcenter = getRectCenter(p);
        List<Point> list = Arrays.asList(precenter, pcenter);
        List<Point> res = iot.transformPointList(list);
        double dis = getDistance(res.get(0),res.get(1), xratio, yratio);
        return  dis;

    }
    private   double getDistance(Point previousPos, Point netxPos, double xratio, double yratio){

        if(previousPos != null && netxPos != null){
            double pcx = previousPos.x ;
            double pcy = previousPos.y ;
            double ccx = netxPos.x;
            double ccy = netxPos.y ;

            if(pcx < 0 || pcy <0 || ccx < 0 || ccy < 0){
//                System.out.println(previousPos);
//                System.out.println(netxPos);
//                System.out.println("=========================================================");
                logger.warn(key + " has position out of size");
            }
            return  Math.sqrt( ((pcx - ccx) *(pcx -ccx)*xratio*xratio)  +
                    ((pcy - ccy)*(pcy - ccy)*yratio*yratio)   );
        }
        return -1;
    }
    public void markedDetectedLost(ArrayList<Integer> list){
        for (int i = 0; i < list.size(); i++) {
            CarDes car = trackers.get(i);
            car.setMarkedDelete();
            car.setMarkedDelete();
        }

    }
    @Override
    public String toString() {
        return key + ":TrackerList{" +
                "selectedType='" + selectedType + '\'' +
                ", trackers=" + trackers +
                ", startCount=" + startCount +
                '}';
    }
    public String toJson(){

        Gson gson = new Gson();
        /**
         * String toJson(Object src)
         * 将对象转为 json，如 基本数据、POJO 对象、以及 Map、List 等
         * 注意：如果 POJO 对象某个属性的值为 null，则 toJson(Object src) 默认不会对它进行转化
         * 结果字符串中不会出现此属性
         */
        String json = gson.toJson(this);
        return  json;
    }
    public TrackerList fromJson(String data){
        Gson gson = new Gson();
        /**
         *  <T> T fromJson(String json, Class<T> classOfT)
         *  json：被解析的 json 字符串
         *  classOfT：解析结果的类型，可以是基本类型，也可以是 POJO 对象类型，gson 会自动转换
         */
        TrackerList p = gson.fromJson(data, TrackerList.class);
        return p;
    }
}

