import random
import numpy as np
import csv

tasks = ["SLA100A", "SLA100B", "SLA191A", "SLA191B", "SLA201", "SLA291", "SLA303", "SLA304", "SLA394", "SLA449", "SLA451"]
locations = ["Slater 003", "Roman 216", "Loft 206", "Roman 201", "Loft 310", "Beach 201", "Beach 301"]
hours = ["10 AM", "11 AM", "12 PM", "1 PM", "2 PM", "3 PM"]
instructors = ["Lock", "Glen", "Banks", "Richards", "Shaw", "Singer", "Uther", "Tyler", "Numen", "Zeldin"]
location_capacity = {"Slater 003": 45, "Roman 216": 30, "Loft 206": 75, "Roman 201": 50, "Loft 310": 108, "Beach 201": 60, "Beach 301": 75}

population_size = 500
mutation_rate = 0.01
generations = 100

best_fitness_scores = []

def calculate_task_adjustments(individual):
    adjustments = 0

    for task, details in individual.items():
        hour = details["hour"]
        location = details["location"]

        if task in ["SLA100A", "SLA100B"]:
            if individual["SLA100A"]["hour"] != individual["SLA100B"]["hour"]:
                adjustments += 1.0  

        if task in ["SLA191A", "SLA191B"]:
            if individual["SLA191A"]["hour"] != individual["SLA191B"]["hour"]:
                adjustments += 1.0  

        if task == "SLA101":
            if abs(convert_hour_to_number(individual["SLA101A"]["hour"]) - convert_hour_to_number(individual["SLA101B"]["hour"])) > 4:
                adjustments += 0.5 
            if individual["SLA101A"]["hour"] == individual["SLA101B"]["hour"]:
                adjustments -= 0.5
            if abs(convert_hour_to_number(individual["SLA101A"]["hour"]) - convert_hour_to_number(individual["SLA191A"]["hour"])) > 4:
                adjustments += 0.5
                if location not in ["Roman 201", "Beach 201"]:
                    adjustments -= 0.4

        if task == "SLA191":
            if abs(convert_hour_to_number(individual["SLA101A"]["hour"]) - convert_hour_to_number(individual["SLA191A"]["hour"])) == 1:
                adjustments += 0.25  
            if individual["SLA101A"]["hour"] == individual["SLA191A"]["hour"]:
                adjustments -= 0.25

    return adjustments

def convert_hour_to_number(hour):
    hour_mapping = {
        "10 AM": 1,
        "11 AM": 2,
        "12 PM": 3,
        "1 PM": 4,
        "2 PM": 5,
        "3 PM": 6,
    }
    return hour_mapping.get(hour, 0)

def generate_initial_population(population_size, tasks, locations, hours, instructors):
    individuals = []
    for _ in range(population_size):
        schedule = {}
        for task in tasks:
            location = random.choice(locations)
            hour = random.choice(hours)
            instructor = random.choice(instructors)
            schedule[task] = {"location": location, "hour": hour, "instructor": instructor}
        individuals.append(schedule)
    return individuals

def calculate_fitness(individual):
    fitness = 0

    location_count = {}
    hour_count = {}
    instructor_count = {}
    consecutive_time_slots = False

    for task, details in individual.items():
        location = details["location"]
        hour = details["hour"]
        instructor = details["instructor"]

        location_capacity_val = location_capacity.get(location, 0)
        expected_students = get_expected_students(task)

        if expected_students <= location_capacity_val <= 3 * expected_students:
            fitness += 0.3
        elif location_capacity_val > 3 * expected_students:
            fitness -= 0.2
        elif location_capacity_val > 6 * expected_students:
            fitness -= 0.4

        if is_preferred_instructor(task, instructor):
            fitness += 0.6 
        elif is_alternate_instructor(task, instructor):
            fitness += 0.3 
        else:
            fitness -= 0.1

        instructor_count[instructor] = instructor_count.get(instructor, 0) + 1

        if instructor_count[instructor] == 1:
            fitness += 0.3  
        elif instructor_count[instructor] > 1:
            fitness -= 0.2
        if instructor_count[instructor] > 4:
            fitness -= 0.6  
        if instructor_count[instructor] in (1, 2) and instructor != "Dr. Tyler":
            fitness -= 0.5  

        location_count[location] = location_count.get(location, 0) + 1
        hour_count[hour] = hour_count.get(hour, 0) + 1

        fitness += calculate_task_adjustments(individual)

    return fitness

def get_expected_students(task):
    expected_students = {
        "SLA100A": 50,
        "SLA100B": 50,
        "SLA191A": 50,
        "SLA191B": 50,
        "SLA201": 50,
        "SLA291": 50,
        "SLA303": 60,
        "SLA304": 25,
        "SLA394": 20,
        "SLA449": 60,
        "SLA451": 100,
    }
    return expected_students.get(task, 0)

def is_preferred_instructor(task, instructor):
    preferred_instructors = {
        "SLA100A": ["Glen", "Lock", "Banks", "Zeldin"],
        "SLA100B": ["Glen", "Lock", "Banks", "Zeldin"],
        "SLA191A": ["Glen", "Lock", "Banks", "Zeldin"],
        "SLA191B": ["Glen", "Lock", "Banks", "Zeldin"],
        "SLA201": ["Glen", "Banks", "Zeldin", "Shaw"],
        "SLA291": ["Lock", "Banks", "Zeldin", "Singer"],
        "SLA303": ["Glen", "Zeldin", "Banks"],
        "SLA304": ["Glen", "Banks", "Tyler"],
        "SLA394": ["Tyler", "Singer"],
        "SLA449": ["Tyler", "Singer", "Shaw"],
        "SLA451": ["Tyler", "Singer", "Shaw"],
    }
    return instructor in preferred_instructors.get(task, [])

def is_alternate_instructor(task, instructor):
    alternate_instructors = {
        "SLA100A": ["Numen", "Richards"],
        "SLA100B": ["Numen", "Richards"],
        "SLA191A": ["Numen", "Richards"],
        "SLA191B": ["Numen", "Richards"],
        "SLA201": ["Numen", "Richards", "Singer"],
        "SLA291": ["Numen", "Richards", "Shaw", "Tyler"],
        "SLA303": ["Numen", "Singer", "Shaw"],
        "SLA304": ["Numen", "Singer", "Shaw", "Richards", "Uther", "Zeldin"],
        "SLA394": ["Richards", "Zeldin"],
        "SLA449": ["Zeldin", "Uther"],
        "SLA451": ["Zeldin", "Uther", "Richards", "Banks"],
    }
    return instructor in alternate_instructors.get(task, [])

def selection(individuals, fitness_scores):
    selected_parents = []
    num_selected_parents = population_size // 2
    selected_parents_indices = np.argsort(fitness_scores)[-num_selected_parents:]
    selected_parents = [individuals[i] for i in selected_parents_indices]

    return selected_parents

def crossover(parents):
    crossover_point = random.randint(1, len(parents[0]))

    child1 = {}
    child2 = {}

    tasks = list(parents[0].keys())
    for i in range(len(tasks)):
        if i < crossover_point:
            child1[tasks[i]] = parents[0][tasks[i]]
            child2[tasks[i]] = parents[1][tasks[i]]
        else:
            child1[tasks[i]] = parents[1][tasks[i]]
            child2[tasks[i]] = parents[0][tasks[i]]

    return child1, child2

def mutate(individual, mutation_rate):
    mutated_individual = individual.copy()

    for task in individual:
        if random.random() < mutation_rate:
            mutation_type = random.choice(["location", "hour", "instructor"])
            
            if mutation_type == "location":
                mutated_individual[task]["location"] = random.choice(locations)
            elif mutation_type == "hour":
                mutated_individual[task]["hour"] = random.choice(hours)
            elif mutation_type == "instructor":
                mutated_individual[task]["instructor"] = random.choice(instructors)

    return mutated_individual

individuals = generate_initial_population(population_size, tasks, locations, hours, instructors)

best_fitness = -float('inf') 
best_individual = None

for generation in range(generations):
    fitness_scores = [calculate_fitness(individual) for individual in individuals]

    if max(fitness_scores) > best_fitness:
        best_fitness = max(fitness_scores)
        best_individual = individuals[np.argmax(fitness_scores)]
        
    best_fitness_scores.append(best_fitness)
    print(f"Generation {generation + 1} - Best Fitness: {best_fitness}")

    parents = selection(individuals, fitness_scores)

    offspring = []
    for i in range(0, len(parents), 2):
        child1, child2 = crossover(parents[i:i+2])
        offspring.append(child1)
        offspring.append(child2)
    for i in range(len(offspring)):
        offspring[i] = mutate(offspring[i], mutation_rate)

    individuals = offspring

print("Best Schedule:")
for task, details in best_individual.items():
    print(f"{task} - Location: {details['location']}, Hour: {details['hour']}, Instructor: {details['instructor']}")
print(f"Best Fitness Score: {best_fitness}")

with open("best_fitness_scores_report.csv", "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Generation", "Best Fitness Score"])
    for generation, fitness_score in enumerate(best_fitness_scores):
        csv_writer.writerow([generation + 1, fitness_score])

with open("best_schedule.txt", "w") as file:
    file.write("Best Schedule:\n")
    for task, details in best_individual.items():
        file.write(f"{task} - Location: {details['location']}, Hour: {details['hour']}, Instructor: {details['instructor']}\n")
