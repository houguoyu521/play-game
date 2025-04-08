import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from mp_api.client import MPRester
import time
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
import plotly.express as px
import plotly.graph_objects as go


# 请替换为你自己的 Material Project API 密钥
API_KEY = "bLi1JNyQysy3R4OdR9DZYj1beCkK1mru"

# 从 Material Project 爬取数据
def fetch_materials_data(elements, num_data):
    try:
        with MPRester(API_KEY) as mpr:
            progress_bar = st.progress(0)
            status_text = st.empty()

            if num_data == "全部":
                all_docs = []
                chunk_size = 1000
                offset = 0
                while True:
                    docs = mpr.materials.summary.search(elements=elements, num_chunks=chunk_size, offset=offset)
                    if not docs:
                        break
                    all_docs.extend(docs)
                    offset += chunk_size
            else:
                num_data = int(num_data)
                all_docs = mpr.materials.summary.search(elements=elements, num_chunks=num_data)

            # 手动筛选数据
            if num_data != "全部":
                all_docs = all_docs[:num_data]

            total_docs = len(all_docs)
            data = []
            for i, doc in enumerate(all_docs):
                entry = {
                    "formula": doc.formula_pretty,
                    "energy_per_atom": doc.energy_per_atom,
                    "formation_energy_per_atom": doc.formation_energy_per_atom,
                    "band_gap": doc.band_gap
                }
                data.append(entry)
                progress = (i + 1) / total_docs
                progress_bar.progress(progress)
                status_text.text(f"正在搜索材料数据: {int(progress * 100)}%")
            status_text.text("材料数据搜索完成！")

            df = pd.DataFrame(data)
            return df
    except Exception as e:
        st.error(f"在从 Material Project 获取数据时发生错误: {e}")
        return pd.DataFrame()


# 处理 formula 转换为特征值
def convert_formula_to_features(data):
    if 'formula' in data.columns:
        st.write("正在将 formula 转换为特征值...")
        # 将字符串化学式转换为 Composition 对象
        stc = StrToComposition()
        data = stc.featurize_dataframe(data, "formula")
        # 添加元素属性特征
        ep_feat = ElementProperty.from_preset(preset_name="magpie")
        data = ep_feat.featurize_dataframe(data, col_id="composition")
        st.write("formula 转换为特征值完成。")
    return data


# 模拟自动机器学习功能
def auto_ml_function(data, predict_variable):
    if data.empty:
        st.warning("未获取到有效数据，请检查输入的元素。")
        return None, None, None
    try:
        # 筛选出数值列
        numeric_columns = data.select_dtypes(include=['number']).columns
        if predict_variable not in numeric_columns:
            st.error(f"需要预测值 {predict_variable} 不是数值类型，请重新选择。")
            return None, None, None

        X = data[numeric_columns.drop(predict_variable)]
        y = data[predict_variable]

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 机器学习进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        model = RandomForestRegressor()

        # 模拟训练进度
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
            status_text.text(f"模型训练中: {i + 1}%")

        status_text.text("模型训练完成！")
        progress_bar.empty()

        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        return rmse, y_test, y_pred, model.feature_importances_, X.columns
    except Exception as e:
        st.error(f"模型训练过程中出现错误: {e}")
        return None, None, None


def main():
    # 初始化会话状态
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'elements' not in st.session_state:
        st.session_state.elements = []
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame()
    if 'predict_variable' not in st.session_state:
        st.session_state.predict_variable = None
    if 'num_data' not in st.session_state:
        st.session_state.num_data = "100"

    st.title("自动机器学习演示")
    st.markdown("此应用允许你输入材料元素名字，从 Material Project 爬取数据，运行自动机器学习模型，并查看模型的准确率。")

    if st.session_state.step == 1:
        # 用户输入材料元素名字
        elements_input = st.text_input("输入想要计算的材料元素名字，用逗号分隔（例如：Fe,O）",
                                       ','.join(st.session_state.elements))
        st.session_state.elements = [element.strip() for element in elements_input.split(",") if element.strip()]

        # 用户选择提取的数据量
        st.session_state.num_data = st.selectbox("选择提取的数据量", ["100", "200", "500", "全部"])

        if st.session_state.elements:
            if st.button("从 Material Project 爬取数据"):
                st.session_state.step = 2

    if st.session_state.step == 2:
        with st.spinner("正在从 Material Project 爬取数据..."):
            st.session_state.data = fetch_materials_data(st.session_state.elements, st.session_state.num_data)
        if not st.session_state.data.empty:
            # 转换 formula 为特征值
            st.session_state.data = convert_formula_to_features(st.session_state.data)

            st.write("数据基本信息：")
            st.session_state.data.info()

            st.write("数据前几行信息：")
            st.dataframe(st.session_state.data.head())

            # 选择需要预测值
            numeric_columns = st.session_state.data.select_dtypes(include=['number']).columns
            st.session_state.predict_variable = st.selectbox("选择需要预测值", numeric_columns)

            if st.session_state.predict_variable:
                if st.button("运行自动机器学习"):
                    st.session_state.step = 3

    if st.session_state.step == 3:
        rmse, y_test, y_pred, feature_importances, feature_names = auto_ml_function(st.session_state.data,
                                                                                    st.session_state.predict_variable)
        if rmse is not None:
            st.write("自动机器学习模型均方根误差：")
            st.metric(label="均方根误差", value=f"{rmse:.2f}")

            # 绘制预测值与真实值的散点图
            fig1 = px.scatter(x=y_test, y=y_pred, labels={'x': '真实值', 'y': '预测值'},
                              title='预测值与真实值的散点图')
            fig1.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                      mode='lines', name='完美预测线'))
            st.plotly_chart(fig1)

            # 绘制特征重要性柱状图
            feature_importance_df = pd.DataFrame({
                '特征': feature_names,
                '重要性': feature_importances
            })
            feature_importance_df = feature_importance_df.sort_values(by='重要性', ascending=False)
            fig2 = px.bar(feature_importance_df, x='特征', y='重要性', title='特征重要性柱状图')
            st.plotly_chart(fig2)

        st.session_state.step = 1  # 完成后回到第一步


if __name__ == "__main__":
    main()
