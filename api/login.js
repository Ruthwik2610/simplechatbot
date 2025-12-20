import users from './users.json'; 
export default function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    const { email, password } = req.body;

    // Search for the user in the JSON file
    const user = users.find(u => u.email === email && u.password === password);

    if (user) {
        // Found! Return success (but don't send the password back)
        return res.status(200).json({
            success: true,
            user: {
                email: user.email,
                name: user.name
            }
        });
    } else {
        // Not found
        return res.status(401).json({ error: 'Invalid email or password' });
    }
}
