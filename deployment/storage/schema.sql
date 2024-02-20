-- Create the users table
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    api_key VARCHAR(50) NOT NULL UNIQUE,
    user_name VARCHAR(100)
);

-- Create the models table
CREATE TABLE models (
    model_id INT PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL UNIQUE,
    model_price DECIMAL(10, 2) NOT NULL,
    model_version VARCHAR(100) NOT NULL
);

-- Create the transactions table
CREATE TABLE queries (
    query_id INT PRIMARY KEY,
    image_base64 TEXT NOT NULL,
    user_id INT REFERENCES users(user_id),
    model_id INT REFERENCES models(model_id)
);
