# import plotly.graph_objects as go
# from sklearn.decomposition import PCA

# def visualize_e8_projected(points):
#     pca = PCA(n_components=2)
#     projected_points = pca.fit_transform(points)

#     trace = go.Scatter(
#         x=projected_points[:,0], y=projected_points[:,1],
#         mode='markers',
#         marker=dict(size=3, color='blue')
#     )

#     fig = go.Figure(data=[trace])
#     fig.update_layout(title="E8 Root System Projection")
#     fig.show()
pass
