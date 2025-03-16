const express = require("express");
const axios = require("axios");
const cors = require("cors");
const bodyParser = require("body-parser");

const app = express();
app.use(cors());
app.use(bodyParser.json());

const FLASK_API_URL = "http://127.0.0.1:5000/query";

app.post("/query", async (req, res) => {
    try {
        const { query } = req.body;
        if (!query) {
            return res.status(400).json({ error: "Query parameter is required" });
        }

        const response = await axios.post(FLASK_API_URL, { query });
        res.json(response.data);
    } catch (error) {
        console.error("Error:", error.message);
        res.status(500).json({ error: "Failed to fetch response from Flask API" });
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Node.js API running on http://127.0.0.1:${PORT}`);
});