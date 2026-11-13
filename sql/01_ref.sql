-- Référentiels à charger AVANT le hackathon (si l'écriture en base est autorisée).
CREATE SCHEMA IF NOT EXISTS ref;

CREATE TABLE IF NOT EXISTS ref.rating_scale (
    notch      smallint PRIMARY KEY,   -- 1 = AAA ... 22 = D
    sp         text NOT NULL,
    moodys     text NOT NULL,
    fitch      text NOT NULL,
    grade      text NOT NULL           -- IG ou HY
);

INSERT INTO ref.rating_scale (notch, sp, moodys, fitch, grade) VALUES
 (1,'AAA','Aaa','AAA','IG'),   (2,'AA+','Aa1','AA+','IG'),  (3,'AA','Aa2','AA','IG'),
 (4,'AA-','Aa3','AA-','IG'),   (5,'A+','A1','A+','IG'),     (6,'A','A2','A','IG'),
 (7,'A-','A3','A-','IG'),      (8,'BBB+','Baa1','BBB+','IG'),(9,'BBB','Baa2','BBB','IG'),
 (10,'BBB-','Baa3','BBB-','IG'),(11,'BB+','Ba1','BB+','HY'), (12,'BB','Ba2','BB','HY'),
 (13,'BB-','Ba3','BB-','HY'),  (14,'B+','B1','B+','HY'),    (15,'B','B2','B','HY'),
 (16,'B-','B3','B-','HY'),     (17,'CCC+','Caa1','CCC+','HY'),(18,'CCC','Caa2','CCC','HY'),
 (19,'CCC-','Caa3','CCC-','HY'),(20,'CC','Ca','CC','HY'),    (21,'C','C','C','HY'),
 (22,'D','D','D','HY')
ON CONFLICT (notch) DO NOTHING;

CREATE TABLE IF NOT EXISTS ref.seniority_rank (
    seniority  text PRIMARY KEY,
    rank       smallint NOT NULL        -- 0 = le plus protégé
);

INSERT INTO ref.seniority_rank (seniority, rank) VALUES
 ('SECURED',0), ('SENIOR_PREFERRED',1), ('SENIOR_UNSECURED',1),
 ('SENIOR_NON_PREFERRED',2), ('TIER2',3), ('SUBORDINATED',3),
 ('AT1',4), ('HYBRID',4)
ON CONFLICT (seniority) DO NOTHING;
