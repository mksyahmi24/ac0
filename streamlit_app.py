import random
import numpy as np
import pandas as pd
import streamlit as st

# Function to load data from a CSV file
def load_data(file):
    """Loads data from a CSV file and returns a list of values from the first column."""
    try:
        df = pd.read_csv(file)
        return df.iloc[:, 0].tolist()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []

# Heuristic information: Inverse of conflicts for a given assignment
def heuristic_value(course, instructor, classroom, timeslot):
    return 1.0

# Fitness function
def fitness(schedule):
    instructor_conflicts = len(schedule) - len(set((c[1], c[3]) for c in schedule))
    room_conflicts = len(schedule) - len(set((c[2], c[3]) for c in schedule))
    return instructor_conflicts + room_conflicts

# Construct a solution
def construct_solution(courses, instructors, classrooms, timeslots, pheromone, alpha, beta):
    solution = []
    for course in courses:
        probabilities = []
        for instructor in instructors:
            for classroom in classrooms:
                for timeslot in timeslots:
                    pheromone_value = pheromone[
                        courses.index(course)][instructors.index(instructor)][classrooms.index(classroom)][timeslots.index(timeslot)]
                    heuristic = heuristic_value(course, instructor, classroom, timeslot)
                    probabilities.append((course, instructor, classroom, timeslot, pheromone_value ** alpha * heuristic ** beta))

        # Normalize probabilities and select an assignment
        total = sum(p[4] for p in probabilities)
        probabilities = [(p[0], p[1], p[2], p[3], p[4] / total) for p in probabilities]
        chosen = random.choices(probabilities, weights=[p[4] for p in probabilities], k=1)[0]
        solution.append((chosen[0], chosen[1], chosen[2], chosen[3]))

    return solution

# Main Streamlit app
st.title("Timetable Optimization using ACO")

with st.sidebar:
    st.header("Upload Your CSV Files")
    students_file = st.file_uploader("Upload Students CSV", type=["csv"], key="students")
    instructors_file = st.file_uploader("Upload Instructors CSV", type=["csv"], key="instructors")
    courses_file = st.file_uploader("Upload Courses CSV", type=["csv"], key="courses")
    classrooms_file = st.file_uploader("Upload Classrooms CSV", type=["csv"], key="classrooms")
    timeslots_file = st.file_uploader("Upload Timeslots CSV", type=["csv"], key="timeslots")

    st.header("ACO Parameters")
    num_ants = st.number_input("Number of Ants", min_value=1, value=5, step=1)  # Reduced number of ants
    num_iterations = st.number_input("Number of Iterations", min_value=1, value=10, step=1)  # Reduced number of iterations
    alpha = st.slider("Pheromone Importance (Alpha)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    beta = st.slider("Heuristic Importance (Beta)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
    evaporation_rate = st.slider("Evaporation Rate", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    Q = st.number_input("Pheromone Update Constant (Q)", min_value=1, value=20, step=1)  # Lower Q for faster computation

if st.button("Run ACO"):
    if not all([students_file, instructors_file, courses_file, classrooms_file, timeslots_file]):
        st.error("Please upload all required CSV files.")
    else:
        # Load data
        students = load_data(students_file)
        instructors = load_data(instructors_file)
        courses = load_data(courses_file)[:10]  # Limit problem size
        classrooms = load_data(classrooms_file)[:3]
        timeslots = load_data(timeslots_file)[:4]

        # Initialize pheromone matrix
        pheromone = np.ones((len(courses), len(instructors), len(classrooms), len(timeslots)))

        # Main ACO loop
        best_solution = None
        best_fitness = float('inf')

        progress_bar = st.progress(0)
        for iteration in range(num_iterations):
            solutions = []
            solution_fitness = []

            # Generate solutions with ants
            for _ in range(num_ants):
                solution = construct_solution(courses, instructors, classrooms, timeslots, pheromone, alpha, beta)
                solutions.append(solution)
                solution_fitness.append(fitness(solution))

            # Update best solution
            min_fitness_index = np.argmin(solution_fitness)
            if solution_fitness[min_fitness_index] < best_fitness:
                best_solution = solutions[min_fitness_index]
                best_fitness = solution_fitness[min_fitness_index]

            # Update pheromones
            pheromone *= (1 - evaporation_rate)  # Evaporation

            for solution, fit in zip(solutions, solution_fitness):
                if fit > 0:  # Avoid division by zero
                    for assignment in solution:
                        course_idx = courses.index(assignment[0])
                        instructor_idx = instructors.index(assignment[1])
                        classroom_idx = classrooms.index(assignment[2])
                        timeslot_idx = timeslots.index(assignment[3])
                        pheromone[course_idx][instructor_idx][classroom_idx][timeslot_idx] += Q / fit

            st.text(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")
            progress_bar.progress((iteration + 1) / num_iterations)

        # Display results
        st.success(f"Optimal Fitness Value: {best_fitness}")
        st.subheader("Optimal Timetable:")
        timetable_df = pd.DataFrame(best_solution, columns=["Course", "Instructor", "Classroom", "Timeslot"])
        st.dataframe(timetable_df)
