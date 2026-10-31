-- ============================================================================
-- schema.sql
-- Run this against the SAME MySQL server your Flask app connects to.
-- Safe to re-run: it widens existing columns, creates missing tables.
-- ============================================================================

-- 1. Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE db;

-- ============================================================================
-- 2. user3  — application users
-- ============================================================================
CREATE TABLE IF NOT EXISTS user3 (
    id          INT PRIMARY KEY AUTO_INCREMENT,
    name        VARCHAR(50)  NOT NULL,
    email       VARCHAR(100) NOT NULL UNIQUE,
    password    VARCHAR(255) NOT NULL,          -- werkzeug scrypt hashes are ~162 chars
    created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Widen password on a pre-existing table (fixes error 1406 / "Data too long")
ALTER TABLE user3
    MODIFY COLUMN password VARCHAR(255) NOT NULL;

-- Widen name/email defensively in case an older build had them smaller
ALTER TABLE user3
    MODIFY COLUMN name  VARCHAR(50)  NOT NULL,
    MODIFY COLUMN email VARCHAR(100) NOT NULL;

-- ============================================================================
-- 3. history  — prediction history
-- ============================================================================
CREATE TABLE IF NOT EXISTS history (
    id                INT PRIMARY KEY AUTO_INCREMENT,
    user_email        VARCHAR(100) NULL,
    source            VARCHAR(30)  NOT NULL,
    identifier        VARCHAR(255) NOT NULL,
    prediction_label  TINYINT      NOT NULL,
    probability       DECIMAL(6,4) NOT NULL,
    created_at        DATETIME     NOT NULL,
    CONSTRAINT fk_history_user
        FOREIGN KEY (user_email)
        REFERENCES user3(email)
        ON DELETE SET NULL
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- 4. Verification — these SELECTs must show what's in the comments
-- ============================================================================

-- Expect: password | varchar(255) | NO |  | NULL |
SHOW COLUMNS FROM user3 LIKE 'password';

-- Expect: name | varchar(50) ... and email | varchar(100) ...
SHOW COLUMNS FROM user3 WHERE Field IN ('name', 'email');

-- Expect: 255 | 255 | 255
SELECT
    MAX(CASE WHEN COLUMN_NAME = 'password' THEN CHARACTER_MAXIMUM_LENGTH END) AS password_len,
    MAX(CASE WHEN COLUMN_NAME = 'name'     THEN CHARACTER_MAXIMUM_LENGTH END) AS name_len,
    MAX(CASE WHEN COLUMN_NAME = 'email'    THEN CHARACTER_MAXIMUM_LENGTH END) AS email_len
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME   = 'user3'
  AND COLUMN_NAME IN ('password', 'name', 'email');

-- Expect: 2  (user3 + history)
SELECT COUNT(*) AS tables_present
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME   IN ('user3', 'history');