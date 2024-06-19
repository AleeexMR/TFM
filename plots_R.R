################################################################################
#                   Code for graphs scenarios
###############################################################################

library(igraph)

#edges for each scenario
edges_s1 <- matrix(c(1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10), ncol = 2, byrow = TRUE)
edges_s2 <- matrix(c(1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 11, 12, 12, 13, 13, 14,14,15), ncol = 2, byrow = TRUE)
edges_s3 <- matrix(c(1, 2, 2, 3, 3, 4, 5, 6, 5, 7, 5, 8, 5, 9, 5, 10, 6, 7, 6, 8, 6, 9, 6, 10, 7, 8, 7, 9, 7, 10, 8, 9, 8, 10, 9, 10), ncol = 2, byrow = TRUE)
edges_s4 <- matrix(c(1, 2, 2, 3, 3, 4, 5, 6, 5, 7, 5, 8, 5, 9, 5, 10, 6, 7, 6, 8, 6, 9, 6, 10, 7, 8, 7, 9, 7, 10, 8, 9, 8, 10, 9, 10, 11, 12, 12, 13, 13, 14, 14, 15), ncol = 2, byrow = TRUE)
edges_s5 <- matrix(c(1, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4, 5, 6, 5, 7, 5, 8, 5, 9, 5, 10, 6, 7, 6, 8, 6, 9, 6, 10, 7, 8, 7, 9, 7, 10, 8, 9, 8, 10, 9, 10), ncol = 2, byrow = TRUE)
edges_s6 <- matrix(c(1, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4, 5, 6, 5, 7, 5, 8, 5, 9, 5, 10, 6, 7, 6, 8, 6, 9, 6, 10, 7, 8, 7, 9, 7, 10, 8, 9, 8, 10, 9, 10, 11, 12, 12, 13, 13, 14, 14, 15), ncol = 2, byrow = TRUE)
edges_s7 <- matrix(c(1, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4, 5, 6, 5, 7, 5, 8, 5, 9, 5, 10, 6, 7, 6, 8, 6, 9, 6, 10, 7, 8, 7, 9, 7, 10, 8, 9, 8, 10, 9, 10, 11, 12, 11, 13, 11, 14, 11, 15, 12, 13, 12, 14, 12, 15, 13, 14, 13, 15, 14, 15), ncol = 2, byrow = TRUE)

# Create graph objects from edge lists, ensuring all 15 vertices are present
create_graph <- function(edges) {
  g <- graph_from_edgelist(edges, directed = FALSE)
  g <- add_vertices(g, 15 - vcount(g))
  g
}

graph_s1 <- create_graph(edges_s1)
graph_s2 <- create_graph(edges_s2)
graph_s3 <- create_graph(edges_s3)
graph_s4 <- create_graph(edges_s4)
graph_s5 <- create_graph(edges_s5)
graph_s6 <- create_graph(edges_s6)
graph_s7 <- create_graph(edges_s7)

# function to plot the different networks
plot_network <- function(graph, title, edge_color) {
  plot(graph, vertex.label = 1:15, layout = layout_in_circle, 
       edge.arrow.size = 0.5, vertex.size = 25, vertex.label.cex = 0.8,
       vertex.color = "skyblue", edge.color = "gray20", main = "")
  mtext(title, side = 3, line = -4, cex = 1)
}

par(mfrow = c(1, 3), mar = c(2, 2, 2, 2))  # Adjust layout for plot the number
#of graphs that u want, i use mfrow =c(1,3) in the thesis

plot_network(graph_s1, "Scenario 1")
plot_network(graph_s2, "Scenario 2")
plot_network(graph_s3, "Scenario 3")

plot_network(graph_s4, "Scenario 4")
plot_network(graph_s5, "Scenario 5")
plot_network(graph_s6, "Scenario 6")

plot_network(graph_s7, "Scenario 7")

plot.new()


########################################################################################
####                 Code for the 3 tsp plots
########################################################################################

library(ggplot2)
library(TSP)
library(ggrepel)

set.seed(12345)  # For reproducibility
nodes <- data.frame(
  x = runif(12, 0, 100),
  y = runif(12, 0, 100)
)

#distance matrix
distance_matrix <- as.matrix(dist(nodes))

# suboptimal TSP solution by using a heuristic method
tsp <- TSP(distance_matrix)
suboptimal_solution <- solve_TSP(tsp, method = "nearest_insertion")
suboptimal_tour <- as.integer(suboptimal_solution)
suboptimal_tour_nodes <- nodes[suboptimal_tour, ]
suboptimal_tour_nodes <- rbind(suboptimal_tour_nodes, suboptimal_tour_nodes[1, ])  # To make a complete loop

p1 <- ggplot(data = nodes, aes(x = x, y = y)) +
  geom_point(size = 11) +  # Increase point size
  geom_path(data = suboptimal_tour_nodes, aes(x = x, y = y), color = "red", size = 1.2) + 
  geom_text_repel(aes(label = 1:12), size = 9, nudge_x = 4, nudge_y = 3) +  
  theme_minimal() +
  theme(
    plot.title = element_blank(),
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank()
  )

print(p1)

# Destroy some edges (destroy function)
destroy_indices <- c(2, 5)  # Indices of the edges to be destroyed
destroyed_suboptimal_tour_nodes <- suboptimal_tour_nodes
destroyed_suboptimal_tour_nodes[c(destroy_indices, destroy_indices + 1), ] <- NA
destroyed_suboptimal_tour_nodes$x[7] <- NA
destroyed_suboptimal_tour_nodes$x[2] <- 50.922434
destroyed_suboptimal_tour_nodes$y[2] <- 95.1658754
destroyed_suboptimal_tour_nodes$x[3] <- 3.453544
destroyed_suboptimal_tour_nodes$y[3] <- 96.5415323

#plot the TSP solution with destroyed edges
p2 <- ggplot(data = nodes, aes(x = x, y = y)) +
  geom_point(size = 11) +  # Increase point size
  geom_path(data = destroyed_suboptimal_tour_nodes, aes(x = x, y = y), color = "red", size = 1.2, na.rm = TRUE) +  # Increase path size
  geom_text_repel(aes(label = 1:12), size = 9, nudge_x = 4, nudge_y = 3) +  
  theme_minimal() +
  theme(
    plot.title = element_blank(),
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank()
  )

print(p2)

# Finally apply the repair function = Solve the TSP to find the optimal solution
optimal_solution <- solve_TSP(tsp)
optimal_tour <- as.integer(optimal_solution)
optimal_tour_nodes <- nodes[optimal_tour, ]
optimal_tour_nodes <- rbind(optimal_tour_nodes, optimal_tour_nodes[1, ])  # To make a complete loop

# Plot the optimal TSP solution
p3 <- ggplot(data = nodes, aes(x = x, y = y)) +
  geom_point(size = 11) +  # Increase point size
  geom_path(data = optimal_tour_nodes, aes(x = x, y = y), color = "red", size = 1.2) +  #Increase path size
  geom_text_repel(aes(label = 1:12), size = 9, nudge_x = 4, nudge_y = 3) +  #adjust position of text labels
  theme_minimal() +
  theme(
    plot.title = element_blank(),
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank()
  )

print(p3)


#######################################################################################
#   code for the visual representation of our destroy and repair functions
######################################################################################

library(ggplot2)
library(dplyr)
library(tidyr)

#To obtain all the plots of the thesis, i only modify the grid_data variable

grid_data <- expand.grid(x = 1:9, y = 1:3) %>%
  mutate(formula = case_when(
    x == 1 & y == 3 ~ "0.6",
    x == 2 & y == 3 ~ "0.2",
    x == 3 & y == 3 ~ "0.2",
    x == 4 & y == 3 ~ "0.6",
    x == 5 & y == 3 ~ "0.2",
    x == 6 & y == 3 ~ "0.2",
    x == 7 & y == 3 ~ "0.6",
    x == 8 & y == 3 ~ "0.2",
    x == 9 & y == 3 ~ "0.2",
    
    x == 1 & y == 2 ~ "0.2",
    x == 2 & y == 2 ~ "0.6",
    x == 3 & y == 2 ~ "0.2",
    x == 4 & y == 2 ~ "0.2",
    x == 5 & y == 2 ~ "0.6",
    x == 6 & y == 2 ~ "0.2",
    x == 7 & y == 2 ~ "0.2",
    x == 8 & y == 2 ~ "0.6",
    x == 9 & y == 2 ~ "0.2",
    
    x == 1 & y == 1 ~ "0.2",
    x == 2 & y == 1 ~ "0.2",
    x == 3 & y == 1 ~ "0.6",
    x == 4 & y == 1 ~ "0.2",
    x == 5 & y == 1 ~ "0.2",
    x == 6 & y == 1 ~ "0.6",
    x == 7 & y == 1 ~ "0.2",
    x == 8 & y == 1 ~ "0.2",
    x == 9 & y == 1 ~ "0.6",
    TRUE ~ "0"
  ))

# Regions with different colors
grid_data <- grid_data %>%
  mutate(region = case_when(
    x <= 9 & y == 3 ~ "soft_red",
    x <=9 & y == 2 ~ "soft_green",
    x <= 9 & y ==1 ~ "soft_blue",
    TRUE ~ "pink"
  ))

ggplot(grid_data, aes(x = x, y = y, fill = region)) +
  geom_tile(color = "black", size=1.5) +
  geom_text(aes(label = formula), size = 9, color = "black", parse = TRUE) +
  scale_fill_manual(values = c("soft_red" = "#f96e50", 
                               "soft_green" = "#90de7d", 
                               "soft_blue" = "#77baf8", 
                               "pink" = "pink")) +
  theme_minimal() +
  theme(axis.text = element_blank(),
        axis.title = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank(),
        legend.position = "none",
        plot.title = element_blank())