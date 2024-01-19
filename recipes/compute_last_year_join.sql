SELECT 
    "cust_web_joined"."customer_id" AS "customer_id",
    "cust_web_joined"."pages_visited" AS "pages_visited",
    "cust_web_joined"."campaign" AS "campaign",
    "cust_web_joined"."ip" AS "ip",
    "cust_web_joined"."Country" AS "Country",
    "cust_web_joined"."GDP_per_cap" AS "GDP_per_cap",
    "crm_last_year_sql"."birth" AS "birth",
    "crm_last_year_sql"."price_first_item_purchased" AS "price_first_item_purchased",
    "crm_last_year_sql"."gender" AS "gender",
    "crm_last_year_sql"."revenue" AS "revenue"
  FROM "PUBLIC"."node-82567e9f_MODERNONLINECUSTOMERLIFETIMEVALUE_CUST_WEB_JOINED" "cust_web_joined"
  LEFT JOIN "PUBLIC"."node-82567e9f_MODERNONLINECUSTOMERLIFETIMEVALUE_CRM_LAST_YEAR_SQL" "crm_last_year_sql"
    ON "cust_web_joined"."customer_id" = "crm_last_year_sql"."customer_id"
    
    
    
