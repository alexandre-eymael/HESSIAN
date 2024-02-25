-- Users
INSERT INTO users (user_id, api_key, user_name) VALUES
(1, 'd8c7cbcb-447d-4273-b285-5c88626b23be', 'Alexandre Eymael'),
(2, '08cf40e6-0033-4a44-bc99-702760fb1606', 'Louis Colson'),
(3, '7c268438-ef90-4bed-894f-6e63a1beda1c', 'Badei Alrahel');

-- Models
INSERT INTO models (model_id, model_name, model_price, model_version) VALUES
(1, 'small', 0.02, '1.0.0'),
(2, 'base',  0.04, '1.0.0'),
(3, 'large', 0.05, '1.0.0');

-- Queries start empty