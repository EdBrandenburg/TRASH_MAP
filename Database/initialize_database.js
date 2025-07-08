const sqlite3 = require('sqlite3').verbose();
const fs = require('fs');
const csv = require('csv-parser');

const db = new sqlite3.Database('./database_factory.db');

db.serialize(() => {
  ///////////////////////////////////////// Create de tables //////////////////////////////////

  db.run(`CREATE TABLE IF NOT EXISTS Users (
    username TEXT PRIMARY KEY NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    nb_pic INTEGER DEFAULT 0,
    role TEXT NOT NULL CHECK (role IN ('admin', 'user', 'anonyme')) DEFAULT 'user',
    points REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );`);

  
  db.run(`CREATE TABLE IF NOT EXISTS Images (
    file_path TEXT PRIMARY KEY,
    label TEXT NOT NULL CHECK(label IN ('clean', 'dirty')),
    true_label TEXT NOT NULL CHECK(true_label IN ('clean', 'dirty','?')),
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    username TEXT NOT NULL,
    localisation TEXT default 'unknown',
    description TEXT default 'no description',
    Quizz TEXT NOT NULL CHECK(label IN ('clean', 'dirty','pachyderm')) default 'pachyderm',
    FOREIGN KEY (username) REFERENCES Users(username) ON DELETE CASCADE
  );`);

  /////////////////////////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////// Prepare les requêtes //////////////////////////////////
  const insertImageStmt = db.prepare(`INSERT INTO Images 
    (file_path, label, true_label, username)
    VALUES (?, ?, ?, ?)`);

  const insertUserStmt = db.prepare(`INSERT INTO Users 
    (username, email, password, nb_pic, role, points, created_at)
    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)`);



  /////////////////////////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////// Function close  //////////////////////////////////
  let doneCount = 0;

  function checkDone() {
    doneCount++;
    if (doneCount === 2) {
      console.log("✅ Import terminé !");
      db.all("SELECT COUNT(*) AS count FROM Images", (_, rows) => {
        console.log(`🖼️ Total images insérées : ${rows[0].count}`);
      });
      db.all("SELECT COUNT(*) AS count FROM Users", (_, rows) => {
        console.log(`👤 Total utilisateurs insérés : ${rows[0].count}`);
      });
      db.close();
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////// Importer les données //////////////////////////////////


  fs.createReadStream('../Without ML/features_with_enhanced_pattern_labels.csv')
    .pipe(csv())
    .on('data', (row) => {
        const trueLabel = (row.true_label === '' || row.true_label === undefined) ? '?' : row.true_label;
      insertImageStmt.run(
        row.file,
        row.auto_label,
        trueLabel,
        'anonymous',
        (err) => {
          if (err && err.code === 'SQLITE_CONSTRAINT') {
            console.warn(`⚠️ Doublon ou contrainte échouée : ${row.file}`);
            console.error('❌ Erreur image :', err.message);
          } else {
            console.log(`✅ Image : ${row.file}`);
          }
        }
      );
    })
    .on('end', () => {
      insertImageStmt.finalize();
      checkDone();
    });

  // Importer les utilisateurs
  fs.createReadStream('./users.csv')
    .pipe(csv())
    .on('data', (row) => {
      insertUserStmt.run(
        row.username,
        row.email,
        row.password,
        parseInt(row.nb_pic),
        row.role,
        parseFloat(row.points),
        (err) => {
          if (err) {
            console.error(`❌ Erreur utilisateur : ${row.username}`, err.message);
          } else {
            console.log(`✅ Utilisateur : ${row.username}`);
          }
        }
      );
    })
    .on('end', () => {
      insertUserStmt.finalize();
      checkDone();
    });
});
///////////////////////////////////////////////////////////////////////////////////////////////
