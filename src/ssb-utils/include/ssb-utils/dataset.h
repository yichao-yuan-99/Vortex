#pragma once

namespace ssb {

constexpr unsigned long long c_total_num = 30000000ull;
constexpr unsigned long long s_total_num = 2000000ull;
constexpr unsigned long long p_total_num = 2000000ull;
constexpr unsigned long long d_total_num = 2556ull;
constexpr unsigned long long lo_total_num = 5999989813ull;

constexpr char c_custkey_f[] = "CUSTOMER0";
constexpr char c_city_f[]    = "CUSTOMER3";
constexpr char c_nation_f[]  = "CUSTOMER4";
constexpr char c_region_f[]  = "CUSTOMER5";

constexpr char s_suppkey_f[] = "SUPPLIER0";
constexpr char s_city_f[]    = "SUPPLIER3";
constexpr char s_nation_f[]  = "SUPPLIER4";
constexpr char s_region_f[]  = "SUPPLIER5";

constexpr char p_partkey_f[]  = "PART0";
constexpr char p_mfgr_f[]     = "PART2";
constexpr char p_category_f[] = "PART3";
constexpr char p_brand1_f[]   = "PART4";

constexpr char d_datekey_f[]          = "DDATE0";
constexpr char d_year_f[]             = "DDATE4";
constexpr char d_yearmonthnum_f[]     = "DDATE5";

constexpr char lo_custkey_f[]       = "LINEORDER2";
constexpr char lo_partkey_f[]       = "LINEORDER3";
constexpr char lo_suppkey_f[]       = "LINEORDER4";
constexpr char lo_orderdate_f[]     = "LINEORDER5";
constexpr char lo_quantity_f[]      = "LINEORDER8";
constexpr char lo_extendedprice_f[] = "LINEORDER9";
constexpr char lo_discount_f[]      = "LINEORDER11";
constexpr char lo_revenue_f[]       = "LINEORDER12";
constexpr char lo_supplycost_f[]    = "LINEORDER13";

struct SSBDataSubsetConfig {
  bool c_custkey = false;
  bool c_city = false;
  bool c_nation = false;
  bool c_region = false;

  bool s_suppkey = false;
  bool s_city = false;
  bool s_nation = false;
  bool s_region = false;

  bool p_partkey = false;
  bool p_mfgr = false;
  bool p_category = false;
  bool p_brand1 = false;

  bool d_datekey = false;
  bool d_year = false;
  bool d_yearmonthnum = false;

  bool lo_custkey = false;
  bool lo_partkey = false;
  bool lo_suppkey = false;
  bool lo_orderdate = false;
  bool lo_quantity = false;
  bool lo_extendedprice = false;
  bool lo_discount = false;
  bool lo_revenue = false;
  bool lo_supplycost = false;
};

constexpr SSBDataSubsetConfig Q1xDataConfig {
  false, // c_custkey
  false, // c_city
  false, // c_nation
  false, // c_region

  false, // s_suppkey
  false, // s_city
  false, // s_nation
  false, // s_region

  false, // p_partkey
  false, // p_mfgr
  false, // p_category
  false, // p_brand1

  false, // d_datekey
  false, // d_year
  false, // d_yearmonthnum

  false, // lo_custkey
  false, // lo_partkey
  false, // lo_suppkey
  true,  // lo_orderdate
  true,  // lo_quantity
  true,  // lo_extendedprice
  true,  // lo_discount
  false, // lo_revenue
  false  // lo_supplycost
};

constexpr SSBDataSubsetConfig Q21DataConfig {
  false, // c_custkey
  false, // c_city
  false, // c_nation
  false, // c_region

  true,  // s_suppkey
  false, // s_city
  false, // s_nation
  true,  // s_region

  true,  // p_partkey
  false, // p_mfgr
  true,  // p_category
  true,  // p_brand1

  true,  // d_datekey
  true,  // d_year
  false, // d_yearmonthnum

  false, // lo_custkey
  true,  // lo_partkey
  true,  // lo_suppkey
  true,  // lo_orderdate
  false, // lo_quantity
  false, // lo_extendedprice
  false, // lo_discount
  true,  // lo_revenue
  false  // lo_supplycost
};

constexpr SSBDataSubsetConfig Q22DataConfig {
  false, // c_custkey
  false, // c_city
  false, // c_nation
  false, // c_region

  true,  // s_suppkey
  false, // s_city
  false, // s_nation
  true,  // s_region

  true,  // p_partkey
  false, // p_mfgr
  false, // p_category
  true,  // p_brand1

  true,  // d_datekey
  true,  // d_year
  false, // d_yearmonthnum

  false, // lo_custkey
  true,  // lo_partkey
  true,  // lo_suppkey
  true,  // lo_orderdate
  false, // lo_quantity
  false, // lo_extendedprice
  false, // lo_discount
  true,  // lo_revenue
  false  // lo_supplycost
};

constexpr SSBDataSubsetConfig Q23DataConfig = Q22DataConfig;

constexpr SSBDataSubsetConfig Q31DataConfig {
  true,  // c_custkey
  false, // c_city
  true,  // c_nation
  true,  // c_region

  true,  // s_suppkey
  false, // s_city
  true,  // s_nation
  true,  // s_region

  false,  // p_partkey
  false,  // p_mfgr
  false,  // p_category
  false,  // p_brand1

  true,  // d_datekey
  true,  // d_year
  false, // d_yearmonthnum

  true,  // lo_custkey
  false, // lo_partkey
  true,  // lo_suppkey
  true,  // lo_orderdate
  false, // lo_quantity
  false, // lo_extendedprice
  false, // lo_discount
  true,  // lo_revenue
  false  // lo_supplycost
};

constexpr SSBDataSubsetConfig Q32DataConfig {
  true,  // c_custkey
  true,  // c_city
  true,  // c_nation
  false, // c_region

  true,  // s_suppkey
  true,  // s_city
  true,  // s_nation
  false, // s_region

  false,  // p_partkey
  false,  // p_mfgr
  false,  // p_category
  false,  // p_brand1

  true,  // d_datekey
  true,  // d_year
  false, // d_yearmonthnum

  true,  // lo_custkey
  false, // lo_partkey
  true,  // lo_suppkey
  true,  // lo_orderdate
  false, // lo_quantity
  false, // lo_extendedprice
  false, // lo_discount
  true,  // lo_revenue
  false  // lo_supplycost
};

constexpr SSBDataSubsetConfig Q33DataConfig {
  true,  // c_custkey
  true,  // c_city
  false, // c_nation
  false, // c_region

  true,  // s_suppkey
  true,  // s_city
  false, // s_nation
  false, // s_region

  false,  // p_partkey
  false,  // p_mfgr
  false,  // p_category
  false,  // p_brand1

  true,  // d_datekey
  true,  // d_year
  false, // d_yearmonthnum

  true,  // lo_custkey
  false, // lo_partkey
  true,  // lo_suppkey
  true,  // lo_orderdate
  false, // lo_quantity
  false, // lo_extendedprice
  false, // lo_discount
  true,  // lo_revenue
  false  // lo_supplycost
};

constexpr SSBDataSubsetConfig Q34DataConfig {
  true,  // c_custkey
  true,  // c_city
  false, // c_nation
  false, // c_region

  true,  // s_suppkey
  true,  // s_city
  false, // s_nation
  false, // s_region

  false,  // p_partkey
  false,  // p_mfgr
  false,  // p_category
  false,  // p_brand1

  true,  // d_datekey
  true,  // d_year
  true,  // d_yearmonthnum

  true,  // lo_custkey
  false, // lo_partkey
  true,  // lo_suppkey
  true,  // lo_orderdate
  false, // lo_quantity
  false, // lo_extendedprice
  false, // lo_discount
  true,  // lo_revenue
  false  // lo_supplycost
};

constexpr SSBDataSubsetConfig Q41DataConfig {
  true,  // c_custkey
  false, // c_city
  true,  // c_nation
  true,  // c_region

  true,  // s_suppkey
  false, // s_city
  false, // s_nation
  true,  // s_region

  true,   // p_partkey
  true,   // p_mfgr
  false,  // p_category
  false,  // p_brand1

  true,  // d_datekey
  true,  // d_year
  false, // d_yearmonthnum

  true,  // lo_custkey
  true,  // lo_partkey
  true,  // lo_suppkey
  true,  // lo_orderdate
  false, // lo_quantity
  false, // lo_extendedprice
  false, // lo_discount
  true,  // lo_revenue
  true   // lo_supplycost
};

constexpr SSBDataSubsetConfig Q42DataConfig {
  true,  // c_custkey
  false, // c_city
  true,  // c_nation
  true,  // c_region

  true,  // s_suppkey
  false, // s_city
  true,  // s_nation
  true,  // s_region

  true,   // p_partkey
  true,   // p_mfgr
  true,   // p_category
  false,  // p_brand1

  true,  // d_datekey
  true,  // d_year
  false, // d_yearmonthnum

  true,  // lo_custkey
  true,  // lo_partkey
  true,  // lo_suppkey
  true,  // lo_orderdate
  false, // lo_quantity
  false, // lo_extendedprice
  false, // lo_discount
  true,  // lo_revenue
  true   // lo_supplycost
};

constexpr SSBDataSubsetConfig Q43DataConfig {
  true,  // c_custkey
  false, // c_city
  false, // c_nation
  true,  // c_region

  true,  // s_suppkey
  true, // s_city
  true,  // s_nation
  false,  // s_region

  true,   // p_partkey
  false,  // p_mfgr
  true,   // p_category
  true,   // p_brand1

  true,  // d_datekey
  true,  // d_year
  false, // d_yearmonthnum

  true,  // lo_custkey
  true,  // lo_partkey
  true,  // lo_suppkey
  true,  // lo_orderdate
  false, // lo_quantity
  false, // lo_extendedprice
  false, // lo_discount
  true,  // lo_revenue
  true   // lo_supplycost
};



} // namespace crystal