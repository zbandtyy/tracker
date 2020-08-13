package detectmotion;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import detectmotion.detector.CascadeDetectCar;
import detectmotion.detector.DetectCar;
import detectmotion.detector.YoloDetectCar;
import detectmotion.interestarea.IOTTransform;
import detectmotion.interestarea.NULLTransform;
import detectmotion.interestarea.PerspectiveConversion;
import detectmotion.tuple.Tuple;
import detectmotion.tuple.Tuple2;
import detectmotion.tuple.Tuple3;
import detectmotion.utils.RectCompute;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import spark.adapter.InheritanceAdapter;

import java.io.IOException;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.*;
import static org.opencv.core.Core.FONT_HERSHEY_SIMPLEX;

/**
 * @author ：tyy
 * @date ：Created in 2020/4/29 1:07
 * @description：使用KCF作为跟踪器进行跟踪，维护一个TrackerList 并且对总的统计数据进行管理更新,只要该类会修改TrackerList的值,或者获取该类的值
 * @modified By：
 * @version: $
 */
@Slf4j
public class OpencvMultiTracker implements Serializable {
    private static final Logger logger = log;

    @Getter
    PerspectiveConversion iot ;
    @Getter
    private   TrackerList trackers;
    @Getter
    private   DetectCar detector = null;
    private  transient  String key = null;
    public  static  final int DEFAULT_LOST_TIME = 1;

    public void setKey(String key) {
        this.key = key;
        trackers.setKey(key);

    }



    public OpencvMultiTracker(String jsonName) throws IOException {
        logger.info(key +": creare json TRANSFORM" + jsonName);
        iot = new IOTTransform(jsonName);
        detector = new YoloDetectCar();
        trackers= new TrackerList();
    }

    public OpencvMultiTracker(Size picSize,Size realSize) {
        logger.info(key +": creare NULL TRANSFORM");
        iot = new NULLTransform(picSize,realSize);
        logger.info(String.valueOf(iot));
        detector = new YoloDetectCar();
        trackers= new TrackerList();
    }



    public OpencvMultiTracker(){
        logger.warn(key+":new OpencvMultiTracker()");
    }
    /***
     *
     * @param predictedObjtects  预测的目标
     * @param dectedObjects  检测到的目标
     * @param MatchedPreictObjects 已经与检测对象匹配的跟踪对象，是(tracker的编号，trackr的位置)组成的键值对
     * @return  预测的目标中  需要在原有tracker中进行更新的对象
     * NOTE:dectedObjects 会在内部做删除已经匹配的处理
     */
    private Map<Integer, Rect2d> findSameObjectByArea(List<Rect2d> predictedObjtects, List<Rect2d> dectedObjects, Map<Integer, Rect2d> MatchedPreictObjects){

        if(predictedObjtects.size() <= 0 || dectedObjects.size() <= 0){
            return  null;
        }
        if(MatchedPreictObjects == null) {
            MatchedPreictObjects = new HashMap<Integer, Rect2d>();       //数据采用的哈希表结构
        }
        logger.debug(key + ":一开始检测到的对象" + dectedObjects.size());
        logger.debug(key + ":预测的对象" + predictedObjtects.size());

        //  3.在预测的位置中 找一个与  检测的结果中心点最近的一个，表明是统一车辆
        ArrayList<Rect2d> unew = new ArrayList<>();
        for (Rect2d pRect:dectedObjects) {//以检测的结果为准
            Integer maxIndex = -1;
            double maxArea = -1;
            for (int i = 0; i < predictedObjtects.size(); i++) {
                double area = RectCompute.getOverlappedArea(pRect,predictedObjtects.get(i));
                //策略1  找重合面积最大的
                if( area > 10 && maxArea < area  ){//预测位置与现有的位置重合位置  最大
                    maxArea = area;
                    maxIndex = i;
                }
            }
            //确定是与上一帧目标相同的patch ,
            if (maxArea > 0 && MatchedPreictObjects.get(maxIndex)  == null ) {
                logger.debug(key + ":detected" + pRect);
                logger.debug(key + ":MaxArea" + maxArea);
                logger.debug(key + ":prect" + predictedObjtects.get(maxIndex));
                unew.add(pRect);
                MatchedPreictObjects.put(maxIndex,pRect);//MatchedPreictObjects 存储的是
            }
        }
        logger.info(key + ":重合的对象" + unew.size());
        dectedObjects.removeAll(unew);//detectedObjects中剩下的是检测到的多余的【即新的车辆】
        logger.info(key  +":剩余的预测对象" + predictedObjtects.size());
        logger.info(key +":剩余的检测对象,用来创建新的对象" + dectedObjects.size());
        return  MatchedPreictObjects;
    }

        /***
         *用检测数据修正
         * @param dectedObjects  实际检测的目标位置
         */
    public void correctBounding(Mat frame, List<Rect2d> dectedObjects){
        ArrayList<Rect2d> trackedcarposes = trackers.getTrackedCarsPos();
        if( dectedObjects == null )
            return;
        else  if(trackedcarposes == null || trackedcarposes.size() == 0){
             createTrackerInList(frame,dectedObjects);
            return;
        }else if(dectedObjects.size() <= 0  ){//检测不出来对象 就使用预测对象替代
            return;
        }
        //1.创建新的Tracker，进行跟踪根据重合范围进行第一次更新
        Map<Integer, Rect2d> needAlterTrackerPos = findSameObjectByArea(trackedcarposes, dectedObjects, null);
        trackers.updateTrackPos(frame,needAlterTrackerPos);//对原来的进行位置跟踪更新，重新创建跟踪器进行跟踪【会删除之前维护的所有对象，重新创建】

        //2.(trackedcarposes)还剩下一个上一帧update预测到的，但是没有检测到匹配的,（1）直接删除
        // 【这种情况 (1)可以继续追踪 （2）直接删除，下一次不进行跟踪了】
        // 【2】如果需要这段代码表明，没有检测到也继续进行跟踪这个在updateTrackPos中做了删除处理, 如果detector的检测率不高的话
//        ArrayList<Integer> notMatchedTracker = new ArrayList<>();//上一批跟踪的对象，但是没有被跟踪到的
//        for (int i = 0; i < trackedcarposes.size(); i++) {
//            if(needAlterTrackerPos.get(i) == null){
//                notMatchedTracker.add(i);
//            }
//        }
//
//        reloadTracker(frame,notMatchedTracker);
//        trackers.markedDetectedLost(notMatchedTracker);//标记检测失败，次数未losttime - 1次（如果检测失败，还能容忍一次update失败，就会立即删除）


        //3.创建新的tracker
        createTrackerInList(frame,dectedObjects);//表示为新检测到的新物体,创建新的tracker 会改变内部的tracker的顺序，所以最后运行！！！！！！！


    }
    public void reloadTracker(Mat frame,ArrayList<Integer> indexofreloadlist){//检测不到但是跟踪到了的对象进行保留处理{
        logger.warn(key + ": reload" + indexofreloadlist);
        trackers.setNewTracker(frame,indexofreloadlist);
    }
    private   void saveDetectorStat(){
        trackers.updatePhaseAndnexPoses();
    }
    private   void cleanLost(int losttime){
        trackers.cleanLostedTrackers(losttime);
    }
    public void createTrackerInList(Mat frame,List<Rect2d> willAddCar){
        if(iot != null) {
            ArrayList<Rect2d> deletedRect = new ArrayList<>();
            for (int i = 0; i < willAddCar.size(); i++) {
                Rect2d r = willAddCar.get(i);
                if (!iot.isInsidePicArea(r.tl()) && !iot.isInsidePicArea(r.br())){
                    logger.info(key + ":removed: br" + r.br() +"  tl" +  r.tl());
                    deletedRect.add(r);
                }
            }
            willAddCar.removeAll(deletedRect);
            logger.info(key + ":remove above all rect  size = " + deletedRect.size() +"because not in interested area");
        }
        trackers.createNewTrackersByArea(frame,willAddCar);
    }

    public void trackObjectsofFrame(Mat frame,boolean reload){
        if(reload == true){
            int size = trackers.getTrackers().size();
            ArrayList list = new ArrayList();
            for (int i = 0; i < size; i++) {
                list.add(i);
            }
            reloadTracker(frame,list);
        }
        //1.预测位置
      trackers.update(frame);//更新位置
      if(iot != null) {
          trackers.deletedNotInArea(iot);//删除不在指定范围内的车辆tracker
      }
      cleanLost(DEFAULT_LOST_TIME);//清理丢失的目标，消失的帧数  时间 进行清理,只要发生丢失就进行清理

    }
    //清理所有超过边界的边框
    public  void deleteCarsOutFrame(List<Rect2d> rect2ds,Size frameSize){
        int height = (int) frameSize.height;
        ArrayList<Rect2d> deleted = new ArrayList<>();
        for (Rect2d rect : rect2ds) {
            double rh = ( (rect.y + rect.height) - height);

            if( rh > 0 ){
                deleted.add(rect);
                log.info("删除超过边界的边框" + rect);
                continue;
            }

        }
        rect2ds.removeAll(deleted);
    }

    public void detectAndCorrectObjofFrame(Mat frame){
        if(detector == null){
            return;
        }
        logger.info(key +":==========检测之前==========\n");

        List<Rect2d> dectedObjects = detector.detectObject(frame);
        deleteCarsOutFrame(dectedObjects,frame.size());
        logger.info(key +":==========detected==========\n" + dectedObjects);
        correctBounding(frame, dectedObjects);
        cleanLost(DEFAULT_LOST_TIME);//清理丢失的目标，消失的帧数  时间 进行清理
        saveDetectorStat();//更新状态为 detector阶段
        logger.info(key +":==========检测完成最后剩下的trackers==========\n" + trackers.getTrackers());
        return  ;
    }
    public void drawCarsBoundingBoxAndCount(Mat frame){

        ArrayList<Tuple2<Rect2d,Long>> manyCarsInfo = trackers.getPosAndCount();
        for (Tuple carInfos : manyCarsInfo) {
            Optional<Rect2d> carpos = carInfos._1();
            Rect2d pos = carpos.get();
            Imgproc.rectangle(frame, pos.tl(), pos.br(), new Scalar(0,255,0),2);
            Optional<Long> count = carInfos._2();
            Imgproc.putText(frame, "" +count.get(),
                    new Point(pos.tl().x + pos.width / 2, pos.y - 5)
                    , FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0,255,0), 2);//显示标识
        }


    }

    public void drawCarsSpeed(double time,Mat frame){
        ArrayList<Tuple3<Rect2d, Double,Double>> speedandpos = trackers.getSpeed(time, iot);
        DecimalFormat df = new DecimalFormat("#.00");
        DecimalFormat dflength = new DecimalFormat("#");
        double V = 0;
        double S = 0;
        double L =0;

        for (int i = 0; i < speedandpos.size(); i++) {
            Optional<Rect2d> carpos = speedandpos.get(i)._1();
            Rect2d pos = carpos.get();
            Optional<Double> count = speedandpos.get(i)._2();//m/s
            String str = df.format(count.get()*3.6);
            Imgproc.putText(frame, str +"km/h",
                    new Point(pos.tl().x + pos.width / 2, pos.y - 20)
                    , FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0,0,255), 2);//显示标识
            Optional<Double> carlength = speedandpos.get(i)._3();
//            String str = dflength.format(carlength.get());
//            Imgproc.putText(frame, str,
//                    new Point(pos.tl().x + pos.width / 2, pos.y + 10)
//                    , FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0,0,255), 2);//显示标识
            L = L + carlength.get();
            V = V + count.get();
            S = S +  V*time;
        }
        if(speedandpos.size() > 0) {
            V = V / speedandpos.size();
            S = S / speedandpos.size();
            L = L / speedandpos.size();
            String str= df.format((V * time) / (L + S) );
            Imgproc.putText(frame, "flow" + str,
                    new Point(10, 75 )
                    , FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0), 2);//显示标识
        }

    }

    /***
     *
     * @param frame  需要处理的帧，进行绘制
     * @param time   detectedFramegap每次进行detect的时间gap
     */
    public  void  drawTrackerBox(Mat frame,double time) {

        drawCarsBoundingBoxAndCount( frame);
        if(time == 0)
            return;
        drawCarsSpeed(time,frame);
    }

    public  void  drawStatistic(Mat frame,double batchFPS) {
        if(trackers == null || frame == null){
            return;
        }
        Imgproc.putText(frame, "FPS:" + String.format("%.2f", batchFPS),
                new Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 0, 255), 2);


        Imgproc.putText(frame,"count:" + trackers.getCurTrackersCount(),
                new Point(10,110), FONT_HERSHEY_SIMPLEX, 1, new Scalar(255,0,0), 2);
    }
    //for test
    public  void  drawdetectedBoundingBox(Mat frame ) {
        List<Rect2d> objects = detector.detectObject(frame);
        for (int i = 0; i < objects.size(); i++) {
            Imgproc.rectangle(frame,objects.get(i).tl() , objects.get(i).br(), new Scalar(255, 255, 0),2);
        }
    }

    public String toJson(){

        GsonBuilder builder = new GsonBuilder();
        builder.registerTypeAdapter(DetectCar.class,new InheritanceAdapter<DetectCar>());
        builder.registerTypeAdapter(PerspectiveConversion.class,new InheritanceAdapter<PerspectiveConversion>());
        /**
         * iot中的mtx 和 inversemtx 为地址【如果不合适，可以自己重新设置新的】
         * detector的是地址【如果不合适，可以自己重新设置新的】
         * trackerlist中的trcker是地址 【如果不合适，可以自己重新设置新的】，
         * 目前存储的值是原来的地址，那么问题来了，在多个分组的时候，会改变吗？？？？
         * String toJson(Object src)
         * 将对象转为 json，如 基本数据、POJO 对象、以及 Map、List 等
         * 注意：如果 POJO 对象某个属性的值为 null，则 toJson(Object src) 默认不会对它进行转化
         * 结果字符串中不会出现此属性
         */
        String json = builder.create().toJson(this);
        return  json;
    }
    public  static OpencvMultiTracker fromJson(String data){
        GsonBuilder builder = new GsonBuilder();
        builder.registerTypeAdapter(DetectCar.class,new InheritanceAdapter<DetectCar>());
        builder.registerTypeAdapter(PerspectiveConversion.class,new InheritanceAdapter<PerspectiveConversion>());
        OpencvMultiTracker opencvMultiTracker = builder.create().fromJson(data, OpencvMultiTracker.class);

        //TrackerList trakerlist = opencvMultiTracker.getTrackers();


        /**
         *  <T> T fromJson(String json, Class<T> classOfT)
         *  json：被解析的 json 字符串
         *  classOfT：解析结果的类型，可以是基本类型，也可以是 POJO 对象类型，gson 会自动转换
         */
        return opencvMultiTracker;
    }



}

