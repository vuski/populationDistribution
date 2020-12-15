#include <Rcpp.h>
#include <vector>
#include <omp.h>
#include <unordered_map>
#include <unordered_set>


// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using std::vector;
using namespace Rcpp;
using std::string;
using std::cout;
using std::endl;
using std::unordered_map;
using std::unordered_set;

struct XY {
  double x;
  double y;
};

struct XYINT {
  int x;
  int y;
};

struct Indirect {
  int first;
  int last;
};

struct Bbox {
  double xmin;
  double xmax;
  double ymin;
  double ymax;
};

//250m 그리드보다 작은 그리드로 설정할 경우
//여기서 곱하는 GRIDC 수치나, 자료형 unsigned int 를 64bit형으로 바꿔야 할 수도 있다.
#define GRIDTYPE unsigned int
#define GRIDC 10000

#define GRIDUNIT 50
#define BLDGGRID 250
#define XMIN 745000
#define YMIN 1387000
#define XSIZE 14280

bool pointInPolygon(XYINT& point, vector<vector<XY>>& polygon);
inline GRIDTYPE getGridIndex(int x, int y);
inline void getXYfromGridIndex(GRIDTYPE index, int& x, int& y);
unordered_set<GRIDTYPE> bldgGridSet;
vector<vector<vector<XY>>> admBndry;
vector<Bbox> admBbox;
vector<double> admPopu;
void readAdmBoundary(NumericVector& xvec, NumericVector& yvec,
                     IntegerVector& L1, IntegerVector& L2, IntegerVector& L3,
                     vector<vector<vector<XY>>>& admBndry_,
                     vector<Bbox>& admBbox_);


// [[Rcpp::export]]
void bldgGridToSet(IntegerVector gridx, IntegerVector gridy) 
{
  bldgGridSet.clear();
  
  int count = gridx.size();
  
  for (int i=0 ; i<count ; i++) {
    
    int x = gridx[i];
    int y = gridy[i];
    
    
    GRIDTYPE xy = (GRIDTYPE)(x + y * GRIDC);
    bldgGridSet.insert(xy);
  }
  
  cout << "그리드 입력 끝"<<endl;
  /*
  int idx = 0;
  for (GRIDTYPE xy : bldgGridSet)
  {
    cout << xy << endl;
    if (idx==100) break;
    idx++;
  }
  */
}


// [[Rcpp::export]]
void putAdmPopu(NumericVector popu)
{
  admPopu.clear();
  
  int count = popu.size();
  admPopu.resize(count);
  
  for (int i=0 ; i<count ; i++) admPopu[i] = popu[i];
  cout << "인구 입력 끝"<<endl;
  
}


// [[Rcpp::export]]
void putAdmBoundary(NumericVector xvec, NumericVector yvec,
                    IntegerVector L1, IntegerVector L2, IntegerVector L3)
{
  //행정경계 입력
  //외곽과 구멍의 구분 없이/폐곡선만 구분해서 입력
  //pointInPolygon 알고리즘 상 외곽과 구멍의 구분이 필요 없기 때문
  
  
  admBndry.clear();
  admBbox.clear();
  
  readAdmBoundary(xvec, yvec, L1, L2, L3, admBndry, admBbox);
}


// [[Rcpp::export]]
DataFrame distributeValue(int threads)
{
  
  vector<unordered_map<GRIDTYPE, double>> valueMap(threads);
  
  int i;
  
  //도형 하나씩 돌면서 그리드 생성하고,
  //해당 그리드의 중점이 경계 안에 들어가는지 체크
  //동시에 건물이 있는 그리드에 해당하는지도 체크
  omp_set_num_threads(threads);
#pragma omp parallel
{
#pragma omp for schedule(guided, 5) nowait
  for (i=0 ; i<(int)admBbox.size() ; i++)
  {
    
    int threadNum = omp_get_thread_num();
    
    Bbox& bbox = admBbox[i];
    int xbegin = (int)(bbox.xmin / GRIDUNIT) * GRIDUNIT + (GRIDUNIT/2); //중점으로
    int xend = (int)(bbox.xmax / GRIDUNIT) * GRIDUNIT + (GRIDUNIT/2); //중점으로
    int ybegin = (int)(bbox.ymin / GRIDUNIT) * GRIDUNIT + (GRIDUNIT/2); //중점으로
    int yend = (int)(bbox.ymax / GRIDUNIT) * GRIDUNIT + (GRIDUNIT/2); //중점으로
    //cout << i << " : " << xbegin << "\t" << xend << "\t" << ybegin << "\t" << yend << endl;
    vector<GRIDTYPE> gridvec;
    
    for (int xx = xbegin ; xx<=xend ; xx += GRIDUNIT)
    {
      
      for (int yy = ybegin ; yy<=yend ; yy += GRIDUNIT)
      {
        
        GRIDTYPE forBldgCheck = (GRIDTYPE)xx / BLDGGRID + ((GRIDTYPE)yy / BLDGGRID ) * GRIDC;
        if (bldgGridSet.find(forBldgCheck) == bldgGridSet.end()) continue; //빌딩 없는 지역이면 그냥 건너뜀
         
        XYINT xy = {xx, yy};
        bool isInside = pointInPolygon(xy, admBndry[i]);
        if (isInside) //경계 안쪽의 점이면 목록에 추가
        {
          GRIDTYPE gridIndex = getGridIndex(xx,yy);
          gridvec.push_back(gridIndex);
        }
        
      } //for yy
    } //for xx
    
    //이제 유효한 그리드만 추출되었다.
    //경계에 주어진 총량을 배분한다.
    
    if (gridvec.size()>0)
    {
      double valueDivided = admPopu[i]/gridvec.size(); //균등분해 할당
      for (GRIDTYPE grid : gridvec)
      {
        valueMap[threadNum][grid] += valueDivided;
      }
    } else //만약 경계가 너무 작아서 그리드 점이 하나도 할당되지 않았으면, bbox 중점 기준으로 몽땅 한곳에 할당당
    {
      //cout << "없는 것도 있다\t"  << i << " : " << xbegin << "\t" << xend << "\t" << ybegin << "\t" << yend << endl;
      GRIDTYPE gridIndex = getGridIndex( (int)((bbox.xmin+bbox.xmax)/2) , (int)((bbox.ymin+bbox.ymax)/2));
      valueMap[threadNum][gridIndex] += admPopu[i];
    }
    //break;
    //if (i%100==0) cout << i << endl;
  } //for admBbox
  
}  //omp parallel
  
  cout << "병렬 계산 완료" << endl;
  //reduce  
  unordered_map<GRIDTYPE, double> valueMapFinal;
  for (i=0 ; i<threads ; i++)
  {
    unordered_map<GRIDTYPE, double>& vm = valueMap[i];
    for (auto& kv : vm)
    {
      valueMapFinal[kv.first] += kv.second;
    }
    
  }
  
  
  unsigned int resultSize = valueMapFinal.size();
  
  cout << "reducing map 완료 : " << resultSize << "개의 grid" << endl;
 
  
  NumericVector xxx(resultSize);
  NumericVector yyy(resultSize);
  NumericVector value(resultSize);
  
  int idx = 0;
  for (auto& kv : valueMapFinal)
  {
    GRIDTYPE gridindex = kv.first;
    int x, y;
    getXYfromGridIndex(gridindex, x, y);
    
    xxx[idx] = (double)x + GRIDUNIT/2;
    yyy[idx] = (double)y + GRIDUNIT/2;
    value[idx] = kv.second;
    idx++;
  }
  //cout << "데이터프레임 변환 준비 완료" << endl;
  DataFrame df = DataFrame::create(Named("x") = xxx,
                                   Named("y") = yyy,
                                   Named("value") = value);
  cout << "데이터프레임 변환 완료" << endl;
  
  
  
  
  return df;
}



// [[Rcpp::export]]
DataFrame filteringSggGrid(IntegerVector gridx, IntegerVector gridy, NumericVector valuevec,
                      NumericVector xvec, NumericVector yvec,
                      IntegerVector L1, IntegerVector L2, IntegerVector L3)
{
  
  vector<vector<vector<XY>>> admBndryPart;
  vector<Bbox> admBboxPart;
  
  readAdmBoundary(xvec, yvec, L1, L2, L3, admBndryPart, admBboxPart);
  
  
  //전체 경계 검출
  double xmin = DBL_MAX;
  double ymin = DBL_MAX;
  double xmax = DBL_MIN;
  double ymax = DBL_MIN;
  
  for (Bbox bb : admBboxPart)
  {
    if (bb.xmin < xmin) xmin = bb.xmin;
    if (bb.xmax > xmax) xmax = bb.xmax;
    if (bb.ymin < ymin) ymin = bb.ymin;
    if (bb.ymax > ymax) ymax = bb.ymax;
  }
  
  
  NumericVector xxx;
  NumericVector yyy;
  NumericVector value;
  
  int count = gridx.size();
  
  for (int i=0; i<count ; i++)
  {
    int x = gridx[i];
    int y = gridy[i];
    
    XYINT xy = {x, y};
    
    //일단 bbox로 검증
    if (x<xmin || x>xmax || y<ymin || y>ymax) continue;
    

    for (vector<vector<XY>>& polygon : admBndryPart )
    {
      
      bool isInside = pointInPolygon(xy, polygon);
      if (isInside) { //한번만 안에 들어 있음 검증되면 저장 후 break
        xxx.push_back(x);
        yyy.push_back(y);
        value.push_back(valuevec[i]);
        break;
      }
    }
    
  }
  
  DataFrame df = DataFrame::create(Named("x") = xxx,
                                   Named("y") = yyy,
                                   Named("value") = value);
  
  cout << "경계추출 완료" << endl;
  return(df);
  
 
  
}



void readAdmBoundary(NumericVector& xvec, NumericVector& yvec,
                     IntegerVector& L1, IntegerVector& L2, IntegerVector& L3,
                     vector<vector<vector<XY>>>& admBndry_,
                     vector<Bbox>& admBbox_)
{
  
  int vectorSize = (int)L3.size();
  
  vector<Indirect> indirect;
  
  //일단 보조 인덱스를 생성
  int indexPre = 1; //L3 가 1번부터 시작하므로.
  int index;
  int first = 0;
  int last;
  int i = 0;
  for (  ; i< vectorSize ; i++)
  {
    index= L3[i];
    if (indexPre != index) {
      last = i;
      Indirect id = {first, last};
      indirect.push_back(id);
      first = i;
      indexPre = index;
    }
  }
  //마지막 넣기
  last = i;
  Indirect id = {first, last};
  indirect.push_back(id);
  
  //이제 넣는다.
  admBndry_.resize(indirect.size());
  admBbox_.resize(indirect.size());
  
  
  for ( i=0 ; i< (int)indirect.size() ; i++)
  {
    
    vector<XY> polygonvec;
    int L1pre = L1[0]; //각각의 값이 변하면 다음 폴리곤이 됨
    int L2pre = L2[0];
    
    Indirect& id = indirect[i];
    
    double xmin = DBL_MAX;
    double ymin = DBL_MAX;
    double xmax = DBL_MIN;
    double ymax = DBL_MIN;
    
    for (int j=id.first ; j<id.last ; j++)
    {
      
      int L1now = L1[j];
      int L2now = L2[j];
      
      if (L1now != L1pre || L2now != L2pre)
      {
        admBndry_[i].push_back(polygonvec); //폴리곤 하나씩 넣는다. 외곽이든 구멍이든 상관 없다.
        L1pre = L1now;
        L2pre = L2now;
        polygonvec.clear();
      }
      XY xy = {xvec[j], yvec[j]};
      polygonvec.push_back(xy);
      if (xy.x < xmin) xmin = xy.x;
      if (xy.x > xmax) xmax = xy.x;
      if (xy.y < ymin) ymin = xy.y;
      if (xy.y > ymax) ymax = xy.y;
    }
    //마지막 것 넣기
    admBndry_[i].push_back(polygonvec);
    
    Bbox bbox = {xmin, xmax, ymin, ymax};
    admBbox_[i] = bbox;
  }
  
  cout << "경계 및 bbox 입력 완료 :" << admBbox_.size() << "개의 개체" <<endl;
  
  
}




inline GRIDTYPE getGridIndex(int x, int y)
{
  return (((GRIDTYPE)x - XMIN)/GRIDUNIT) + XSIZE * ( ((GRIDTYPE)y - YMIN)/GRIDUNIT );
}

inline void getXYfromGridIndex(GRIDTYPE index, int& x, int& y)
{
  x = index%XSIZE * GRIDUNIT + XMIN;
  y = index/XSIZE * GRIDUNIT + YMIN;
}

bool pointInPolygon(XYINT& point, vector<vector<XY>>& polygon) {
  
  // ray-casting algorithm based on
  // http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
  // http://bl.ocks.org/bycoffe/5575904
  bool intersect;
  bool inside = false;
  for (int p = 0; p < (int)polygon.size(); p++) {
    int j = polygon[p].size() - 1;
    for (int i = 0; i < (int)polygon[p].size(); j = i++) {
      intersect = ((polygon[p][i].y > point.y) != (polygon[p][j].y > point.y)) && (point.x < (polygon[p][j].x - polygon[p][i].x) * (point.y - polygon[p][i].y) / (polygon[p][j].y - polygon[p][i].y) + polygon[p][i].x);
      if (intersect) inside = !inside;
    }
  }
  return inside;
}


