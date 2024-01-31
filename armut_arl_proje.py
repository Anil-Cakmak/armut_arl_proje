import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#########################
# Veriyi Hazırlama
#########################

# Veri setinin okutulması.
df_ = pd.read_csv("recommendation_systems/Ödev/armut_data.csv")
df = df_.copy()

# ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturulması.
df["Hizmet"] = df.apply(lambda x: str(x["ServiceId"]) + "_" + str(x["CategoryId"]), axis=1)


# Sepetlerin her bir müşterininin aylık aldığı hizmetler olarak unique id'lerle tanımlanması.
df["New_Date"] = df["CreateDate"].apply(lambda x: x[0:7])
df["SepetId"] = df.apply(lambda x: str(x["UserId"]) + "_" + x["New_Date"], axis=1)

#########################
# Birliktelik Kuralları Üretilmesi.
#########################

# Sepet hizmet pivot table’i oluşturulması.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

df = pd.pivot_table(df, values="CategoryId", index="SepetId", columns="Hizmet", aggfunc="count", fill_value=0). \
    map(lambda x: 1 if x > 0 else 0).astype("bool")


# Adım 2: Birliktelik kurallarının oluşturulması.
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# En son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulunulması.


def arl_recommender(rules_df, service_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, service in enumerate(sorted_rules["antecedents"]):
        for j in list(service):
            if j == service_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, "2_0", 1)


